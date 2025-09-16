"""
SODEF training script for adversarial robustness
Stage 2: Train SODEF defense component with pretrained backbone
"""
import torch
import argparse
import yaml
import os

from models.combined import TwoStageTrainer
from training.sodef_trainer import SODEFTrainer
from utils.data_loaders import get_cifar10_loaders


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SODEF defense component')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='Path to SODEF checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if baseline checkpoint exists
    baseline_checkpoint = config['paths']['baseline_checkpoint']
    if not os.path.exists(baseline_checkpoint):
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")
    
    print(f"Loading baseline model from: {baseline_checkpoint}")
    
    # Load baseline model
    baseline_model = TwoStageTrainer.load_baseline_checkpoint(
        baseline_checkpoint,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes']
    )
    
    # Create SODEF model
    sodef_model = TwoStageTrainer.create_sodef_model(
        baseline_model,
        ode_dim=config['sodef']['ode_dim'],
        num_classes=config['model']['num_classes']
    )
    
    print(f"Created SODEF model with ODE dimension: {config['sodef']['ode_dim']}")
    print(f"Total parameters: {sum(p.numel() for p in sodef_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in sodef_model.parameters() if p.requires_grad):,}")
    
    # Setup regularization weights
    sodef_model.regularizer.weight_diag = config['sodef']['regularization']['weight_diag']
    sodef_model.regularizer.weight_offdiag = config['sodef']['regularization']['weight_offdiag']
    sodef_model.regularizer.weight_f = config['sodef']['regularization']['weight_f']
    sodef_model.regularizer.exponent = config['sodef']['regularization']['exponent']
    sodef_model.regularizer.exponent_off = config['sodef']['regularization']['exponent_off']
    sodef_model.regularizer.exponent_f = config['sodef']['regularization']['exponent_f']
    sodef_model.regularizer.trans = config['sodef']['regularization']['trans']
    sodef_model.regularizer.transoffdig = config['sodef']['regularization']['transoffdig']
    sodef_model.regularizer.num_samples = config['sodef']['regularization']['num_samples']
    
    # Create trainer
    trainer = SODEFTrainer(
        model=sodef_model,
        device=device,
        log_interval=config['logging']['log_interval']
    )
    
    # Set loss weights
    trainer.stability_weight = config['training']['stability_weight']
    trainer.classification_weight = config['training']['classification_weight']
    
    # Setup optimizer
    trainer.setup_optimizer(
        ode_lr=config['training']['ode_lr'],
        classifier_lr=config['training']['classifier_lr'],
        ode_eps=config['training']['ode_eps'],
        classifier_eps=config['training']['classifier_eps']
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
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming SODEF training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train SODEF model using two-phase approach
    best_acc, final_acc = trainer.train_full_pipeline(
        train_loader=train_loader,
        test_loader=test_loader,
        stability_epochs=config['training']['stability_epochs'],
        classification_epochs=config['training']['classification_epochs'],
        save_dir=config['paths']['save_dir']
    )
    
    print(f"\nSODEF training completed!")
    print(f"Best accuracy (temp ODE): {best_acc:.2f}%")
    print(f"Final accuracy (full ODE): {final_acc:.2f}%")
    
    # Save final model info
    final_info = {
        'model_arch': config['model']['arch'],
        'ode_dim': config['sodef']['ode_dim'],
        'num_classes': config['model']['num_classes'],
        'best_accuracy_temp': best_acc,
        'final_accuracy_full': final_acc,
        'stability_epochs': config['training']['stability_epochs'],
        'classification_epochs': config['training']['classification_epochs'],
        'regularization_weights': config['sodef']['regularization'],
        'baseline_checkpoint': baseline_checkpoint,
        'config': config
    }
    
    info_path = os.path.join(config['paths']['save_dir'], 'sodef_model_info.yaml')
    with open(info_path, 'w') as f:
        yaml.dump(final_info, f, default_flow_style=False)
    
    print(f"SODEF model info saved to: {info_path}")


if __name__ == '__main__':
    main()