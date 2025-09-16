"""
Baseline ResNet trainer for CIFAR-10
Stage 1: Train standard classifier
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
import os
from tqdm import tqdm


class BaselineTrainer:
    """Trainer for baseline ResNet model"""
    
    def __init__(self, model, device='cuda', log_interval=100):
        self.model = model.to(device)
        self.device = device
        self.log_interval = log_interval
        
        # Metrics tracking
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        self.best_acc = 0.0
        
    def setup_optimizer(self, lr=0.1, momentum=0.9, weight_decay=5e-4, 
                       scheduler_type='step', step_size=50, gamma=0.1):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        if scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=200, eta_min=0
            )
        else:
            self.scheduler = None
            
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def test(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        self.test_accs.append(test_acc)
        
        return test_loss, test_acc
    
    def save_checkpoint(self, epoch, test_acc, save_dir, is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'test_acc': test_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_accs': self.test_accs
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'New best model saved with accuracy: {test_acc:.2f}%')
    
    def train(self, train_loader, test_loader, epochs=200, save_dir='./checkpoints/baseline'):
        """Complete training loop"""
        print(f"Starting baseline training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Test
            test_loss, test_acc = self.test(test_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Check if best model
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, test_acc, save_dir, is_best)
            
            # Print progress
            print(f'Epoch {epoch:3d}: '
                  f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | '
                  f'Best: {self.best_acc:.2f}%')
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.1f} minutes')
        print(f'Best test accuracy: {self.best_acc:.2f}%')
        
        return self.best_acc
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        if load_optimizer and 'scheduler' in checkpoint and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load metrics
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.test_accs = checkpoint['test_accs']
            
        if 'test_acc' in checkpoint:
            self.best_acc = checkpoint['test_acc']
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Test accuracy: {checkpoint.get('test_acc', 'unknown'):.2f}%")
        
        return checkpoint.get('epoch', 0)