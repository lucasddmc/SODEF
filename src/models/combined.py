"""
Combined models: Backbone + SODEF defense
Implements the two-stage architecture where backbone extracts features
and SODEF provides adversarial robustness
"""
import torch
import torch.nn as nn
from .resnet import resnet18_cifar10, resnet34_cifar10
from .sodef import create_sodef_component, create_full_ode_block


class BaselineModel(nn.Module):
    """Standard ResNet classifier for CIFAR-10"""
    
    def __init__(self, arch='resnet18', num_classes=10):
        super(BaselineModel, self).__init__()
        
        if arch == 'resnet18':
            self.model = resnet18_cifar10(num_classes=num_classes)
        elif arch == 'resnet34':
            self.model = resnet34_cifar10(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        """Extract features before classification"""
        return self.model.get_features(x)


class SODEFModel(nn.Module):
    """Complete SODEF model: Backbone + ODE Defense + Classifier"""
    
    def __init__(self, backbone, ode_dim=64, num_classes=10, use_full_ode=False):
        super(SODEFModel, self).__init__()
        
        # Backbone for feature extraction (frozen during SODEF training)
        self.backbone = backbone
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            feature_dim = backbone.get_features(dummy_input).shape[1]
        
        # Create SODEF components
        sodef_components = create_sodef_component(
            feature_dim=feature_dim,
            ode_dim=ode_dim,
            num_classes=num_classes
        )
        
        self.feature_adapter = sodef_components['adapter']
        self.odefunc = sodef_components['odefunc']
        self.classifier = sodef_components['classifier']
        self.regularizer = sodef_components['regularizer']
        
        # ODE block (temp for training, full for inference)
        if use_full_ode:
            self.ode_block = create_full_ode_block(self.odefunc)
        else:
            self.ode_block = sodef_components['ode_block']
            
        self.use_full_ode = use_full_ode
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone.get_features(x)
        
        # Adapt features to ODE dimension
        ode_input = self.feature_adapter(features)
        
        # Apply ODE processing
        ode_output = self.ode_block(ode_input)
        
        # Final classification
        logits = self.classifier(ode_output)
        
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters for SODEF training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def switch_to_full_ode(self):
        """Switch from temp ODE to full integration"""
        if not self.use_full_ode:
            self.ode_block = create_full_ode_block(self.odefunc)
            self.use_full_ode = True
            
    def switch_to_temp_ode(self):
        """Switch from full ODE to temp (training mode)"""
        if self.use_full_ode:
            from .sodef import ODEBlockTemp
            self.ode_block = ODEBlockTemp(self.odefunc)
            self.use_full_ode = False
            
    def get_ode_features(self, x):
        """Get features after ODE processing (for analysis)"""
        features = self.backbone.get_features(x)
        ode_input = self.feature_adapter(features)
        ode_output = self.ode_block(ode_input)
        return ode_output
    
    def compute_stability_loss(self, x):
        """Compute Lyapunov stability regularization loss"""
        features = self.backbone.get_features(x)
        ode_input = self.feature_adapter(features)
        
        device = x.device
        loss, metrics = self.regularizer.compute_loss(
            self.odefunc, ode_input, device
        )
        
        return loss, metrics


class TwoStageTrainer:
    """Helper class for two-stage training process"""
    
    @staticmethod
    def create_baseline_model(arch='resnet18', num_classes=10):
        """Create baseline model for stage 1"""
        return BaselineModel(arch=arch, num_classes=num_classes)
    
    @staticmethod
    def create_sodef_model(baseline_model, ode_dim=64, num_classes=10):
        """Create SODEF model from trained baseline for stage 2"""
        
        # Create SODEF model with the trained backbone
        sodef_model = SODEFModel(
            backbone=baseline_model.model,
            ode_dim=ode_dim,
            num_classes=num_classes,
            use_full_ode=False  # Start with temp ODE for training
        )
        
        # Freeze backbone by default
        sodef_model.freeze_backbone()
        
        return sodef_model
    
    @staticmethod
    def load_baseline_checkpoint(checkpoint_path, arch='resnet18', num_classes=10):
        """Load baseline model from checkpoint"""
        model = BaselineModel(arch=arch, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        return model
    
    @staticmethod
    def save_model(model, path, epoch=None, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        
    @staticmethod
    def load_sodef_checkpoint(checkpoint_path, baseline_model, ode_dim=64, num_classes=10):
        """Load SODEF model from checkpoint"""
        sodef_model = TwoStageTrainer.create_sodef_model(
            baseline_model, ode_dim=ode_dim, num_classes=num_classes
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            sodef_model.load_state_dict(checkpoint['state_dict'])
        else:
            sodef_model.load_state_dict(checkpoint)
            
        return sodef_model


def create_model_pair(baseline_checkpoint=None, arch='resnet18', ode_dim=64, num_classes=10):
    """Convenience function to create both baseline and SODEF models"""
    
    if baseline_checkpoint:
        # Load from checkpoint
        baseline_model = TwoStageTrainer.load_baseline_checkpoint(
            baseline_checkpoint, arch=arch, num_classes=num_classes
        )
    else:
        # Create new baseline
        baseline_model = TwoStageTrainer.create_baseline_model(
            arch=arch, num_classes=num_classes
        )
    
    # Create SODEF model
    sodef_model = TwoStageTrainer.create_sodef_model(
        baseline_model, ode_dim=ode_dim, num_classes=num_classes
    )
    
    return baseline_model, sodef_model