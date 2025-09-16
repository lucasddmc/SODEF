"""
Evaluation script for testing robustness against adversarial attacks
Uses AutoAttack as in the original SODEF paper and adds FGSM/PGD baselines

Key fixes compared to earlier version:
- Ensure attacks operate on images in [0,1] by using an unnormalized DataLoader
    and wrapping the model with CIFAR-10 normalization internally
- Remove torch.no_grad() around adversarial example generation (gradients needed)
- Support AutoAttack custom subsets via attacks_to_run
- Add native FGSM and PGD (L_inf) evaluation
"""
import torch
import torch.nn as nn
import argparse
import yaml
import os
import time
from autoattack import AutoAttack
from utils.data_loaders import get_cifar10_loaders, CIFAR10_MEAN, CIFAR10_STD
from attacks.fgsm_pgd import fgsm_attack, pgd_linf

from models.combined import TwoStageTrainer, create_model_pair


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class NormalizedModel(nn.Module):
    """Wrap a model to apply CIFAR-10 normalization inside the forward.

    This lets us feed [0,1] images to attacks while the wrapped model
    still receives normalized inputs as expected by training.
    """
    def __init__(self, model, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.model = model
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x):
        x = (x - self._mean) / self._std
        return self.model(x)


def evaluate_clean_accuracy(model, test_loader, device):
    """Evaluate clean accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def _concat_loader_to_tensor(loader, device, max_batches=None):
    """Helper to stack a loader into tensors on device."""
    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        xs.append(x)
        ys.append(y)
        if max_batches is not None and i + 1 >= max_batches:
            break
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y


def evaluate_autoattack(model, test_loader, device, epsilon=8/255,
                        norm='Linf', batch_size=100, version='standard', attacks_to_run=None,
                        max_batches=None):
    """Evaluate using AutoAttack on [0,1] images (model must handle normalization)."""
    print(f"Setting up AutoAttack with epsilon={epsilon}, norm={norm}, version={version}")

    # Prepare the full evaluation tensor (optionally limited for speed)
    x_test, y_test = _concat_loader_to_tensor(test_loader, device, max_batches=max_batches)
    print(f"Evaluating on {x_test.shape[0]} test samples...")

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version)
    if version == 'custom' and attacks_to_run:
        adversary.attacks_to_run = attacks_to_run

    start_time = time.time()
    # Important: do not wrap with no_grad, attacks need gradients
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    eval_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        output = model(x_adv)
        _, predicted = output.max(1)
        robust_acc = 100.0 * predicted.eq(y_test).sum().item() / y_test.size(0)

    print(f"AutoAttack evaluation completed in {eval_time/60:.1f} minutes")
    return robust_acc, eval_time


def evaluate_fgsm_pgd(model, test_loader, device, attack_name, epsilon=8/255,
                       alpha=2/255, steps=10, batch_size=100, max_batches=None):
    """Evaluate robustness under FGSM/PGD (L_inf) on [0,1] images.

    model is expected to apply normalization internally (use NormalizedModel).
    Returns: robust accuracy (%), total_eval_time (sec)
    """
    assert attack_name in {"FGSM", "PGD"}
    model.eval()

    total_correct = 0
    total = 0
    start_time = time.time()

    for i, (x, y) in enumerate(test_loader):
        if max_batches is not None and i + 1 > max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        if attack_name == "FGSM":
            x_adv = fgsm_attack(model, x, y, epsilon)
        else:
            x_adv = pgd_linf(model, x, y, eps=epsilon, alpha=alpha, steps=steps, random_start=True)

        with torch.no_grad():
            out = model(x_adv)
            pred = out.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += y.size(0)

    eval_time = time.time() - start_time
    robust_acc = 100.0 * total_correct / max(1, total)
    return robust_acc, eval_time


def evaluate_model(model_path, model_type, config, device):
    """Evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} model")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    # Load data
    # Important: for attacks, we load UNNORMALIZED images in [0,1]
    # and wrap the model with an internal normalization layer.
    _, test_loader, _ = get_cifar10_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['eval']['batch_size'],
        num_workers=config['data']['num_workers'],
        download=True,
        normalize=False
    )
    
    # Load model
    if model_type == 'baseline':
        model = TwoStageTrainer.load_baseline_checkpoint(
            model_path, 
            arch=config['model']['arch'],
            num_classes=config['model']['num_classes']
        )
    elif model_type == 'sodef':
        # Load baseline first, then SODEF
        baseline_model = TwoStageTrainer.load_baseline_checkpoint(
            config['eval']['baseline_checkpoint'],
            arch=config['model']['arch'],
            num_classes=config['model']['num_classes']
        )
        model = TwoStageTrainer.load_sodef_checkpoint(
            model_path,
            baseline_model,
            ode_dim=config['sodef']['ode_dim'],
            num_classes=config['model']['num_classes']
        )
        # Ensure using full ODE for evaluation
        model.switch_to_full_ode()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move and wrap with normalization for evaluation
    model = model.to(device)
    norm_model = NormalizedModel(model).to(device)
    
    # Evaluate clean accuracy
    print("Evaluating clean accuracy...")
    clean_acc = evaluate_clean_accuracy(norm_model, test_loader, device)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    
    # Evaluate adversarial robustness
    results = {'clean_accuracy': clean_acc}
    
    for attack_config in config['eval']['attacks']:
        print(f"\nEvaluating {attack_config['name']} attack...")
        name = attack_config['name']
        if name.upper().startswith('FGSM'):
            robust_acc, eval_time = evaluate_fgsm_pgd(
                norm_model, test_loader, device, attack_name='FGSM',
                epsilon=attack_config['epsilon'],
                batch_size=config['eval']['batch_size'],
                max_batches=attack_config.get('max_batches')
            )
        elif name.upper().startswith('PGD'):
            robust_acc, eval_time = evaluate_fgsm_pgd(
                norm_model, test_loader, device, attack_name='PGD',
                epsilon=attack_config['epsilon'],
                alpha=attack_config.get('alpha', 2/255),
                steps=attack_config.get('steps', 10),
                batch_size=config['eval']['batch_size'],
                max_batches=attack_config.get('max_batches')
            )
        else:
            robust_acc, eval_time = evaluate_autoattack(
                norm_model, test_loader, device,
                epsilon=attack_config['epsilon'],
                norm=attack_config['norm'],
                batch_size=config['eval']['batch_size'],
                version=attack_config.get('version', 'standard'),
                attacks_to_run=attack_config.get('attacks_to_run'),
                max_batches=attack_config.get('max_batches')
            )

        print(f"{attack_config['name']} robust accuracy: {robust_acc:.2f}%")
        results[f"{attack_config['name']}_robust_accuracy"] = robust_acc
        results[f"{attack_config['name']}_eval_time"] = eval_time
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SODEF models against adversarial attacks')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, choices=['baseline', 'sodef', 'both'], 
                       default='both', help='Which model to evaluate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Results storage
    all_results = {}
    
    # Evaluate baseline model
    if args.model in ['baseline', 'both']:
        if os.path.exists(config['eval']['baseline_checkpoint']):
            baseline_results = evaluate_model(
                config['eval']['baseline_checkpoint'], 
                'baseline', 
                config, 
                device
            )
            all_results['baseline'] = baseline_results
        else:
            print(f"Baseline checkpoint not found: {config['eval']['baseline_checkpoint']}")
    
    # Evaluate SODEF model
    if args.model in ['sodef', 'both']:
        if os.path.exists(config['eval']['sodef_checkpoint']):
            sodef_results = evaluate_model(
                config['eval']['sodef_checkpoint'], 
                'sodef', 
                config, 
                device
            )
            all_results['sodef'] = sodef_results
        else:
            print(f"SODEF checkpoint not found: {config['eval']['sodef_checkpoint']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()} Model:")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
        
        for key, value in results.items():
            if 'robust_accuracy' in key:
                attack_name = key.replace('_robust_accuracy', '')
                print(f"  {attack_name} Robust Accuracy: {value:.2f}%")
    
    # Save results
    results_dir = config['eval'].get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    import json
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()