"""
SODEF: Stable Neural ODE with Lyapunov-Stable Equilibrium Points
Implementation based on the original paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import geotorch


class ODEFunc(nn.Module):
    """Neural ODE function with Lyapunov-oriented stable MLP dynamics.

    Default layout inspired by the reference implementation:
    ode_dim -> hidden -> hidden -> ode_dim with sin activations and negative coefficient.
    """

    def __init__(self, ode_dim: int, hidden: int = 256, f_coeffi: float = -1.0, act=torch.sin):
        super(ODEFunc, self).__init__()
        self.ode_dim = ode_dim
        self.hidden = hidden
        self.f_coeffi = f_coeffi
        self.act = act
        # Camadas
        self.fc1 = nn.Linear(ode_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, ode_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.f_coeffi * self.fc1(x)
        out = self.act(out)
        out = self.f_coeffi * self.fc2(out)
        out = self.act(out)
        out = self.f_coeffi * self.fc3(out)
        out = self.act(out)
        return out


class ODEBlock(nn.Module):
    """ODE integration block"""
    
    def __init__(self, odefunc, integration_time=5.0, rtol=1e-3, atol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, integration_time]).float()
        self.rtol = rtol
        self.atol = atol
        
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(
            self.odefunc, x, self.integration_time, 
            rtol=self.rtol, atol=self.atol
        )
        return out[1]  # Return final state
    
    @property
    def nfe(self):
        return self.odefunc.nfe
    
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEBlockTemp(nn.Module):
    """Temporary ODE block without integration (for training efficiency)"""
    
    def __init__(self, odefunc):
        super(ODEBlockTemp, self).__init__()
        self.odefunc = odefunc
        
    def forward(self, x):
        # Single step without integration
        return self.odefunc(0, x)
    
    @property
    def nfe(self):
        return self.odefunc.nfe
    
    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class FeatureAdapter(nn.Module):
    """Adapter to project features to ODE dimension with orthogonal constraint.

    Matches the paper/codebase approach using an orthogonal linear map when
    input_dim >= ode_dim. If input_dim < ode_dim, fallback para Linear padrão.
    """

    def __init__(self, input_dim, ode_dim, bias=False):
        super(FeatureAdapter, self).__init__()
        self.input_dim = input_dim
        self.ode_dim = ode_dim
        self.bias = bias

        if input_dim >= ode_dim:
            self.fc = nn.Linear(input_dim, ode_dim, bias=bias)
            # Impõe restrição ortogonal no peso (geotorch)
            geotorch.orthogonal(self.fc, "weight")
        else:
            # Projeção aumento de dimensão sem restrição (não quadrado)
            self.fc = nn.Linear(input_dim, ode_dim, bias=bias)

    def forward(self, x):
        return self.fc(x)


class SODEFClassifier(nn.Module):
    """Final classifier after SODEF processing"""
    
    def __init__(self, ode_dim, num_classes):
        super(SODEFClassifier, self).__init__()
        self.fc = nn.Linear(ode_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)


class LyapunovRegularizer:
    """Lyapunov stability regularization terms"""
    
    def __init__(self, weight_diag=10, weight_offdiag=0, weight_f=0.1,
                 exponent=1.0, exponent_off=0.1, exponent_f=50,
                 trans=1.0, transoffdig=1.0, num_samples=16):
        self.weight_diag = weight_diag
        self.weight_offdiag = weight_offdiag
        self.weight_f = weight_f
        self.exponent = exponent
        self.exponent_off = exponent_off
        self.exponent_f = exponent_f
        self.trans = trans
        self.transoffdig = transoffdig
        self.num_samples = num_samples
        
    def jacobian_regularizer(self, odefunc, z, device):
        """Compute Jacobian-based regularization"""
        regu_diag = 0.
        regu_offdiag = 0.
        
        # Sample random subset for efficiency
        indices = np.random.choice(
            z.shape[0], 
            min(self.num_samples, z.shape[0]), 
            replace=False
        )
        
        for ii in indices:
            # x_i: vetor 1xD
            x_i = z[ii:ii+1, ...]
            D = x_i.shape[1]

            # Jacobiano de f(x) wrt x em x_i
            J = torch.autograd.functional.jacobian(
                lambda x: odefunc(torch.tensor(1.0).to(device), x),
                x_i,
                create_graph=True
            )
            # J tem shape (1,D,1,D) ou similar; vamos achatar para (D,D)
            J = J.view(D, D)

            if J.shape[0] != J.shape[1]:
                raise RuntimeError(f"Jacobian shape is not square: {tuple(J.shape)}")

            # Diagonal (autovalores aproximados via diag de df/dz)
            diag_elements = torch.diagonal(J, 0)
            regu_diag += torch.exp(self.exponent * (diag_elements + self.trans))

            # Off-diagonal: soma das magnitudes fora da diagonal por coluna (ou linha)
            eye = torch.eye(D, device=device)
            off_mask = (1.0 - eye)
            off_abs = torch.abs(J) * off_mask
            off_sum_per_col = off_abs.sum(dim=0)
            regu_offdiag += torch.exp(self.exponent_off * (off_sum_per_col + self.transoffdig))
            
        return regu_diag / self.num_samples, regu_offdiag / self.num_samples
    
    def function_regularizer(self, odefunc, z, device):
        """Regularize function magnitude"""
        f_values = torch.abs(odefunc(torch.tensor(1.0).to(device), z))
        regu_f = torch.pow(self.exponent_f * f_values, 2)
        return regu_f
    
    def compute_loss(self, odefunc, z, device):
        """Compute total regularization loss"""
        regu_diag, regu_offdiag = self.jacobian_regularizer(odefunc, z, device)
        regu_f = self.function_regularizer(odefunc, z, device)
        
        total_loss = (
            self.weight_diag * regu_diag.mean() +
            self.weight_offdiag * regu_offdiag.mean() +
            self.weight_f * regu_f.mean()
        )
        
        return total_loss, {
            'diag_reg': regu_diag.mean().item(),
            'offdiag_reg': regu_offdiag.mean().item(),
            'function_reg': regu_f.mean().item()
        }


def create_sodef_component(feature_dim, ode_dim=64, num_classes=10):
    """Factory function to create SODEF components"""
    
    # Create ODE function
    odefunc = ODEFunc(ode_dim)
    
    # Create feature adapter
    adapter = FeatureAdapter(feature_dim, ode_dim)
    
    # Create ODE block (use temp version for training)
    ode_block = ODEBlockTemp(odefunc)
    
    # Create classifier
    classifier = SODEFClassifier(ode_dim, num_classes)
    
    # Create regularizer
    regularizer = LyapunovRegularizer()
    
    return {
        'odefunc': odefunc,
        'adapter': adapter,
        'ode_block': ode_block,
        'classifier': classifier,
        'regularizer': regularizer
    }


def create_full_ode_block(odefunc, integration_time=5.0):
    """Create full ODE block with integration for inference"""
    return ODEBlock(odefunc, integration_time)