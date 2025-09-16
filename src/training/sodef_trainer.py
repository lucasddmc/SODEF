"""
SODEF trainer for adversarial robustness
Stage 2: Train SODEF defense component with Lyapunov stability
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
import numpy as np


class SODEFTrainer:
    """Trainer for SODEF defense component"""
    
    def __init__(self, model, device='cuda', log_interval=50):
        self.model = model.to(device)
        self.device = device
        self.log_interval = log_interval
        
        # Metrics tracking
        self.stability_losses = []
        self.classification_losses = []
        self.total_losses = []
        self.test_accs = []
        self.best_acc = 0.0
        
        # Loss weights (from original paper)
        self.stability_weight = 1.0
        self.classification_weight = 0.1
        
    def setup_optimizer(self, ode_lr=1e-2, classifier_lr=1e-2, 
                       ode_eps=1e-3, classifier_eps=1e-4):
        """Setup optimizer for different components"""
        
        # Different learning rates for ODE and classifier components
        optimizer_params = [
            {
                'params': [p for name, p in self.model.named_parameters() 
                          if 'odefunc' in name or 'feature_adapter' in name],
                'lr': ode_lr,
                'eps': ode_eps
            },
            {
                'params': [p for name, p in self.model.named_parameters() 
                          if 'classifier' in name],
                'lr': classifier_lr,
                'eps': classifier_eps
            }
        ]
        
        self.optimizer = optim.Adam(optimizer_params, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_stability_phase(self, train_loader, epochs=20, save_dir='./checkpoints/sodef_stability'):
        """Phase 1: Train only stability regularizers (no classification loss)"""
        print(f"Starting SODEF stability training for {epochs} epochs...")
        
        # Disable classifier gradients for this phase
        for param in self.model.classifier.parameters():
            param.requires_grad = False
            
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Stability Epoch {epoch}')
            
            for batch_idx, (data, target) in enumerate(pbar):
                data = data.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Compute stability loss only
                stability_loss, metrics = self.model.compute_stability_loss(data)
                
                stability_loss.backward()
                self.optimizer.step()
                
                running_loss += stability_loss.item()
                
                # Update progress bar
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix({
                        'Stability Loss': f'{stability_loss.item():.6f}',
                        'Diag Reg': f'{metrics["diag_reg"]:.4f}',
                        'OffDiag Reg': f'{metrics["offdiag_reg"]:.4f}',
                        'Func Reg': f'{metrics["function_reg"]:.4f}'
                    })
            
            epoch_loss = running_loss / len(train_loader)
            self.stability_losses.append(epoch_loss)
            
            print(f'Stability Epoch {epoch:3d}: Loss: {epoch_loss:.6f}')
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, 0.0, save_dir, phase='stability')
        
        # Re-enable classifier gradients
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        total_time = time.time() - start_time
        print(f'Stability training completed in {total_time/60:.1f} minutes')
        
    def train_classification_phase(self, train_loader, test_loader, epochs=10, 
                                 save_dir='./checkpoints/sodef_classification'):
        """Phase 2: Fine-tune with classification loss + stability"""
        print(f"Starting SODEF classification training for {epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch_combined(train_loader, epoch)
            
            # Test
            test_acc = self.test(test_loader)
            
            # Check if best model
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
            
            # Save checkpoint
            if epoch % 2 == 0 or is_best:
                self.save_checkpoint(epoch, test_acc, save_dir, phase='classification', is_best=is_best)
            
            print(f'Classification Epoch {epoch:3d}: '
                  f'Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}% | '
                  f'Best: {self.best_acc:.2f}%')
        
        total_time = time.time() - start_time
        print(f'Classification training completed in {total_time/60:.1f} minutes')
        print(f'Best test accuracy: {self.best_acc:.2f}%')
        
        return self.best_acc
    
    def train_epoch_combined(self, train_loader, epoch):
        """Train epoch with both stability and classification losses"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Combined Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Classification loss
            classification_loss = self.criterion(output, target)
            
            # Stability loss
            stability_loss, metrics = self.model.compute_stability_loss(data)
            
            # Combined loss
            total_loss = (self.classification_weight * classification_loss + 
                         self.stability_weight * stability_loss)
            
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'Total': f'{total_loss.item():.4f}',
                    'Class': f'{classification_loss.item():.4f}',
                    'Stab': f'{stability_loss.item():.6f}'
                })
        
        epoch_loss = running_loss / len(train_loader)
        self.total_losses.append(epoch_loss)
        
        return epoch_loss
    
    def test(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        self.test_accs.append(test_acc)
        
        return test_acc
    
    def save_checkpoint(self, epoch, test_acc, save_dir, phase='combined', is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'test_acc': test_acc,
            'stability_losses': self.stability_losses,
            'classification_losses': self.classification_losses,
            'total_losses': self.total_losses,
            'test_accs': self.test_accs,
            'stability_weight': self.stability_weight,
            'classification_weight': self.classification_weight
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_latest_{phase}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(save_dir, f'checkpoint_best_{phase}.pth')
            torch.save(checkpoint, best_path)
            print(f'New best SODEF model saved with accuracy: {test_acc:.2f}%')
    
    def train_full_pipeline(self, train_loader, test_loader, 
                           stability_epochs=20, classification_epochs=10,
                           save_dir='./checkpoints/sodef'):
        """Complete SODEF training pipeline"""
        print("="*60)
        print("SODEF Training Pipeline")
        print("="*60)
        
        # Phase 1: Stability training
        self.train_stability_phase(
            train_loader, 
            epochs=stability_epochs,
            save_dir=os.path.join(save_dir, 'stability')
        )
        
        print("\n" + "="*60)
        
        # Phase 2: Classification training
        best_acc = self.train_classification_phase(
            train_loader, test_loader,
            epochs=classification_epochs,
            save_dir=os.path.join(save_dir, 'classification')
        )
        
        # Switch to full ODE for inference
        print("\nSwitching to full ODE integration for inference...")
        self.model.switch_to_full_ode()
        
        # Final test with full ODE
        final_acc = self.test(test_loader)
        print(f"Final accuracy with full ODE: {final_acc:.2f}%")
        
        # Save final model
        final_checkpoint = {
            'state_dict': self.model.state_dict(),
            'test_acc': final_acc,
            'use_full_ode': True
        }
        torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
        
        return best_acc, final_acc
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load metrics
        if 'stability_losses' in checkpoint:
            self.stability_losses = checkpoint['stability_losses']
        if 'classification_losses' in checkpoint:
            self.classification_losses = checkpoint['classification_losses']
        if 'total_losses' in checkpoint:
            self.total_losses = checkpoint['total_losses']
        if 'test_accs' in checkpoint:
            self.test_accs = checkpoint['test_accs']
            
        if 'stability_weight' in checkpoint:
            self.stability_weight = checkpoint['stability_weight']
        if 'classification_weight' in checkpoint:
            self.classification_weight = checkpoint['classification_weight']
        
        print(f"Loaded SODEF checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Phase: {checkpoint.get('phase', 'unknown')}")
        print(f"Test accuracy: {checkpoint.get('test_acc', 'unknown'):.2f}%")
        
        return checkpoint.get('epoch', 0)