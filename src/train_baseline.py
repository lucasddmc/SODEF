"""
Baseline ResNet training script for CIFAR-10
Stage 1: Train standard classifier that will serve as backbone for SODEF
"""
import torch
import argparse
import yaml
import os

from models.combined import TwoStageTrainer
from training.baseline_trainer import BaselineTrainer
from utils.data_loaders import get_cifar10_loaders


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train baseline ResNet for CIFAR-10')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = TwoStageTrainer.create_baseline_model(
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes']
    )
    
    print(f"Created {config['model']['arch']} model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        device=device,
        log_interval=config['logging']['log_interval']
    )
    
    # Setup optimizer and scheduler
    trainer.setup_optimizer(
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['training']['scheduler']['type'],
        step_size=config['training']['scheduler']['step_size'],
        gamma=config['training']['scheduler']['gamma']
    )
    
    # Load data
    train_loader, test_loader, _ = get_cifar10_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        download=config['data']['download'],
        normalize=config['data']['normalize']
    )
    
    print(f"Data loaded - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train model
    best_acc = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config['training']['epochs'],
        save_dir=config['logging']['save_dir']
    )
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Save final model info
    final_info = {
        'model_arch': config['model']['arch'],
        'num_classes': config['model']['num_classes'],
        'best_accuracy': best_acc,
        'total_epochs': config['training']['epochs'],
        'config': config
    }
    
    info_path = os.path.join(config['logging']['save_dir'], 'model_info.yaml')
    with open(info_path, 'w') as f:
        yaml.dump(final_info, f, default_flow_style=False)
    
    print(f"Model info saved to: {info_path}")


if __name__ == '__main__':
    main()