import argparse
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from models import vgg19, checkpoined_vgg
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, Schedule1F1B
from torchvision import datasets, transforms
from memory import PipelineMemory
import matplotlib.pyplot as plt 
from pprint import pprint
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class DistributedConfig:
    """Configuration for distributed training setup."""
    rank: int
    device: torch.device
    pp_group: dist.ProcessGroup
    stage_index: int
    num_stages: int


def init_distributed() -> DistributedConfig:
    """Initialize distributed training environment.
    
    Returns:
        DistributedConfig: Configuration object containing distributed training setup.
    """
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group()

    # This group can be a sub-group in the N-D parallel case
    pp_group = dist.new_group()
    
    return DistributedConfig(
        rank=rank,
        device=device,
        pp_group=pp_group,
        stage_index=rank,
        num_stages=world_size
    )


def manual_model_split(model: nn.Module, example_input_microbatch: torch.Tensor, 
                      dist_config: DistributedConfig) -> PipelineStage:
    """Split the model into pipeline stages based on the stage index.
    
    Args:
        model: The model to split
        example_input_microbatch: Example input for the model
        dist_config: Distributed training configuration
        
    Returns:
        PipelineStage: The pipeline stage for the current process
    """
    if dist_config.stage_index == 0:
        for i in range(2, model.n_segment):
            del model.segments[str(i)]
            del model.memory_stats[str(i)]
    elif dist_config.stage_index == 1:
        for i in range(2):
            del model.segments[str(i)]
            del model.memory_stats[str(i)]
        for i in range(4, model.n_segment):
            del model.segments[str(i)]
            del model.memory_stats[str(i)]
    else:
        for i in range(4):
            del model.segments[str(i)]
            del model.memory_stats[str(i)]

    return PipelineStage(
        model,
        dist_config.stage_index,
        dist_config.num_stages,
        dist_config.device,
        input_args=example_input_microbatch,
    )


def get_data_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """Create ImageNet data loader with the specified transformations.
    
    Args:
        args: Command line arguments containing data loading parameters
        
    Returns:
        DataLoader: ImageNet training data loader
    """
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    return torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True, 
        sampler=None, 
        drop_last=True
    )


class MemoryProfiler:
    """Handles memory profiling during model training."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_stats: List[Dict[str, Any]] = []
        self.handles = []

    def register_hooks(self, module: nn.Module) -> None:
        """Register memory profiling hooks for the given module."""
        self.handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
        self.handles.append(module.register_forward_hook(self._forward_hook))
        self.handles.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
        self.handles.append(module.register_full_backward_hook(self._backward_hook))

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()

    def _forward_pre_hook(self, module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        self._pre_op_hook(module, 'forward')

    def _forward_hook(self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._post_op_hook(module, args, output, is_forward=True)

    def _backward_pre_hook(self, module: nn.Module, grad_output: tuple[torch.Tensor, ...]) -> None:
        self._pre_op_hook(module, 'backward')

    def _backward_hook(self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        self._post_op_hook(module, grad_input, grad_output, is_forward=False)

    def _pre_op_hook(self, module: nn.Module, direction: str) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_max_memory_allocated(device=self.device)
        
        # Get previous memory usage, falling back to pre_op_memory or 0 if not found
        prev = 0
        if self.memory_stats:
            try:
                last_stat = self.memory_stats[-1]
                # Get previous memory usage, falling back to pre_op_memory or 0 if not found
                prev = last_stat.get('pos_op_memory', last_stat.get('pre_op_memory', 0))
            except:
                pprint(self.memory_stats[-1])
                
        current_memory = torch.cuda.memory_allocated(device=self.device)
        self.memory_stats.append({
            'op': module.__class__.__name__,
            'dir': direction,
            'pre_op_memory': current_memory,
            'diff': current_memory - prev
        })

    def _post_op_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], 
                      outputs: tuple[torch.Tensor, ...] | torch.Tensor, is_forward: bool) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        current_stat = self.memory_stats[-1]
        current_stat['max_op_memory'] = torch.cuda.max_memory_allocated(device=self.device)
        current_stat['pos_op_memory'] = torch.cuda.memory_allocated(device=self.device)
        
        if is_forward:
            outputs_tuple = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            current_stat.update({
                'tensor_input': sum(arg.element_size() * arg.numel() for arg in inputs),
                'tensor_output': sum(out.element_size() * out.numel() for out in outputs_tuple),
                'cuda_output': current_stat['pos_op_memory'] - current_stat['pre_op_memory'],
                'forward_overhead': (current_stat['max_op_memory'] - current_stat['pre_op_memory'] 
                                   - (current_stat['pos_op_memory'] - current_stat['pre_op_memory']))
            })
        else:
            grad_inputs = tuple(x for x in inputs if x is not None)
            current_stat.update({
                'tensor_input_grad': sum(arg.element_size() * arg.numel() for arg in grad_inputs),
                'tensor_output_grad': sum(arg.element_size() * arg.numel() for arg in outputs),
                'cuda_input_grad': current_stat['pos_op_memory'] - current_stat['pre_op_memory'],
                'backward_overhead': (current_stat['max_op_memory'] - current_stat['pre_op_memory'] 
                                    - (current_stat['pos_op_memory'] - current_stat['pre_op_memory']))
            })


def train(schedule: Schedule1F1B, args: argparse.Namespace, dist_config: DistributedConfig,
          train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
          epoch: int, memory_info: PipelineMemory, profile: Optional[int] = None) -> None:
    """Train the model for one epoch.
    
    Args:
        schedule: Pipeline schedule for training
        args: Training arguments
        dist_config: Distributed training configuration
        train_loader: Data loader for training
        optimizer: Optimizer for model parameters
        epoch: Current epoch number
        memory_info: Memory profiling information
        profile: Batch index to profile memory usage (if None, no profiling is done)
    """
    profiler = MemoryProfiler(dist_config.device) if profile is not None else None
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == profile and profiler is not None:
            for seg_id, segment in schedule._stage.submod.segments.items():
                for module in segment.children():
                    profiler.register_hooks(module)

        optimizer.zero_grad()
        
        if dist_config.rank == 0:
            data = data.to(dist_config.device)
            schedule.step(data)
        elif dist_config.rank == dist_config.num_stages - 1:
            target = target.to(dist_config.device)
            losses = []
            output = schedule.step(target=target, losses=losses)
            
            if batch_idx % args.log_interval == 0:
                loss_val = sum(loss.item() for loss in losses) / args.num_microbatches
                print(f'Train Epoch: {epoch} [{batch_idx * len(target)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_val:.6f}')
        else:
            schedule.step()
            
        optimizer.step()

        if batch_idx == profile and profiler is not None:
            profiler.cleanup()
            torch.cuda.memory._dump_snapshot(f"{dist_config.stage_index}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
            memory_info.simulate1F1B(profiler.memory_stats)
            memory_info.simulate1F1B_v(profiler.memory_stats)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='PyTorch Imagenet Pipeline Parallel Training')
    parser.add_argument('data', metavar='DIR', nargs='?', default='/data/imagenet',
                      help='path to dataset (default: imagenet)')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                      help='number of data loading workers (default: 4)')
    parser.add_argument('--num-microbatches', type=int, default=8, metavar='M',
                      help='number of chunks to be split in a mini-batch (default: 8)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                      help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                      help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Initialize model and distributed setup
    model = checkpoined_vgg(vgg19())
    torch.cuda.memory._record_memory_history()
    dist_config = init_distributed()
    
    # Create example input based on stage
    example_input_microbatch = create_example_input(dist_config.stage_index, args)
    
    # Split model and create pipeline stage
    stage = manual_model_split(model, example_input_microbatch, dist_config)
    print(f'Stage: {dist_config.rank}')
    print(stage.submod)

    # Initialize memory profiling
    memory_info = PipelineMemory(
        stage.submod, 
        example_input_microbatch, 
        dist_config.device, 
        dist_config.rank, 
        dist_config.num_stages, 
        args.num_microbatches
    )

    # Create training schedule and optimizer
    schedule = ScheduleGpipe(
        stage, 
        n_microbatches=args.num_microbatches, 
        loss_fn=nn.CrossEntropyLoss()
    )
    optimizer = optim.SGD(stage.submod.parameters(), lr=args.lr)
    train_loader = get_data_loader(args)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(schedule, args, dist_config, train_loader, optimizer, epoch, memory_info, profile=0)

    dist.destroy_process_group()


def create_example_input(stage_index: int, args: argparse.Namespace) -> torch.Tensor:
    """Create example input tensor based on pipeline stage.
    
    Args:
        stage_index: Index of current pipeline stage
        args: Command line arguments containing batch size and microbatch count
        
    Returns:
        torch.Tensor: Example input tensor for the current stage
    """
    batch_size = args.batch_size // args.num_microbatches
    if stage_index == 0:
        return torch.randn(batch_size, 3, 224, 224)
    elif stage_index == 1:
        return torch.randn(batch_size, 256, 28, 28)
    else:
        return torch.randn(batch_size, 512, 7, 7)


if __name__ == '__main__':
    main()
