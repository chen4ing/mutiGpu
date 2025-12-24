import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import matplotlib.pyplot as plt 

class Memory():
    def __init__(
        self, 
        model: nn.Module, 
        example_input_microbatch: torch.Tensor, 
        device: Any, 
    ) -> None:
        self.model = model
        self.example_input_microbatch = example_input_microbatch
        self.device = device

    "list[dict[str, Any]]"
    def profile_memory_stats(self):
        torch.cuda.memory._record_memory_history()
        modules = self.flatten_nn_module([m for _, m in self.model._modules.items()])
        memory_stats = []
        backward_stats = []
        last_memory = [torch.cuda.memory_allocated(device=self.device)]

        handles = []
        for i, module in enumerate(modules):
            def forward_pre_hook(module: nn.Module, args) -> None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated(device=self.device)
                last_memory[0] = torch.cuda.memory_allocated(device=self.device)
            def forward_hook(module: nn.Module, args, output) -> None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if module.__class__.__name__ == 'ReLU':
                    print('input', args[0])
                    print('output', output)
                stat = {}
                stat['op'] = module.__class__.__name__
                stat['input'] = sum(arg.element_size() * arg.numel() for arg in args)
                stat['output'] = output.element_size() * output.numel()
                stat['outputs'] = torch.cuda.memory_allocated(device=self.device) - last_memory[0]
                stat['forward_overhead'] = torch.cuda.max_memory_allocated(device=self.device) - last_memory[0] - stat['outputs']
                memory_stats.append(stat)
            def backward_pre_hook(module: nn.Module, grad_output) -> None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_max_memory_allocated(device=self.device)
                last_memory[0] = torch.cuda.memory_allocated(device=self.device)
            def backward_hook(module: nn.Module, grad_input, grad_output) -> None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                stat = {}
                stat['grad_input'] = sum(grad.element_size() * grad.numel() for grad in grad_input if grad is not None)
                stat['grad_output'] = sum(grad.element_size() * grad.numel() for grad in grad_output)
                stat['grad_inputs'] = torch.cuda.memory_allocated(device=self.device) - last_memory[0]
                stat['backward_overhead'] = torch.cuda.max_memory_allocated(device=self.device) - last_memory[0] - stat['grad_inputs']
                stat['grad_params'] = sum(p.numel() * p.element_size() for p in module.parameters())
                backward_stats.append(stat)

            handles.append(module.register_forward_pre_hook(forward_pre_hook))
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_pre_hook(backward_pre_hook))
            handles.append(module.register_full_backward_hook(backward_hook))

        for p in self.model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)

        out = self.model(self.example_input_microbatch)
        loss = out.sum()
        loss.backward()

        for module in modules:
            module.zero_grad()

        for handle in handles:
            handle.remove()

        for i in range(len(memory_stats)):
            memory_stats[i].update(backward_stats[len(memory_stats) - i - 1])
        torch.cuda.memory._dump_snapshot(f"single.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return memory_stats
    
    " list[nn.Module]"
    def flatten_nn_module(self, module: nn.Module):
        out = []
        for layer in module:
            if isinstance(layer, nn.ModuleList)\
            or isinstance(layer, nn.Sequential):
                out.extend(self.flatten_nn_module(layer))
            else:
                out.append(layer)
        return out

class PipelineMemory():
    def __init__(
        self, 
        submod: nn.Module, 
        example_input_microbatch: torch.Tensor, 
        device: Any, 
        rank: int = 0, 
        world_size: int = 1, 
        num_microbatches: int = 1, 
    ) -> None:
        self.submod = submod
        self.example_input_microbatch = example_input_microbatch
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.num_microbatches = num_microbatches

        # Memory for parameters and gradients.
        self.param_size = sum(p.numel() * p.element_size() for p in self.submod.parameters())
        self.grad_size = self.param_size

        # Build per layer information. 
        self.memory_stats = []
        for segment in submod.memory_stats.values():
            self.memory_stats.append(segment)
        print(self.memory_stats)

        # Memory for the example-input-microbatch-like tensor and input/pred of a minibatch.
        self.empty_tensor_size = self.example_input_microbatch.numel() * self.example_input_microbatch.element_size()
        self.empty_tensor_size += self.memory_stats[-1][-1]['output']
        # Memory for the input/pred of a minibatch.
        self.data_size = self.num_microbatches * self.example_input_microbatch.numel() * self.example_input_microbatch.element_size() if rank == 0 else 0

        # Allocate communication buffer for each micro-batch.
        self.grad_recv_buffer = self.num_microbatches * self.memory_stats[-1][-1]['output'] if self.rank != self.world_size - 1 else 0
        self.act_recv_buffer = self.num_microbatches * self.memory_stats[0][0]['input'] if self.rank != 0 else 0


    def get_static_memory(self) -> int:
        return self.param_size + self.empty_tensor_size + self.data_size + self.grad_recv_buffer + self.act_recv_buffer

    def get_segments_memory_stats(self) -> int:
        return self.memory_stats

    def simulateGpipe(self, cuda_log) -> None:
        world_size = self.world_size
        rank = self.rank
        current_memory = self.get_static_memory()
        def checkpoint_forward(segment):
        