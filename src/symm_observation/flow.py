"""Original joint SymmFlow path plus differentiable ODE integration.

Flow time tau is dimensionless and is NEVER equated to elapsed clinical days.
The exact endpoints here require sigma_min=0, as in the audited three-phase code.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .utils import finite_tensor

@dataclass
class JointPath:
    joint: torch.Tensor
    velocity: torch.Tensor
    tau: torch.Tensor
    noise_x: torch.Tensor
    noise_y: torch.Tensor


def expand_time(t, ref):
    return t.reshape(len(ref), *([1]*(ref.ndim-1)))


def make_path(earlier, later, tau=None, noise_x=None, noise_y=None):
    finite_tensor(earlier, "earlier"); finite_tensor(later, "later")
    if earlier.shape != later.shape or earlier.ndim != 5 or earlier.shape[1] != 24:
        raise ValueError("Paired latents must both have shape [B,24,D,H,W]")
    if tau is None:
        tau = torch.rand(len(earlier), device=earlier.device)
    if tau.shape != (len(earlier),) or not torch.isfinite(tau).all() or ((tau < 0)|(tau > 1)).any():
        raise ValueError("tau must be [B] in [0,1]")
    ex = torch.randn_like(later) if noise_x is None else noise_x
    ey = torch.randn_like(earlier) if noise_y is None else noise_y
    if ex.shape != earlier.shape or ey.shape != earlier.shape:
        raise ValueError("Path noise shapes must match the paired observations")
    finite_tensor(ex, "noise_x"); finite_tensor(ey, "noise_y")
    t = expand_time(tau, earlier)
    x = (1-t)*ex + t*later
    y = (1-t)*earlier + t*ey
    return JointPath(torch.cat((x,y), 1), torch.cat((later-ex, ey-earlier), 1), tau, ex, ey)


def estimate_endpoints(joint, velocity, tau):
    """One-network-evaluation clean endpoint estimates, NOT full ODE samples."""
    x, y = joint.chunk(2, 1)
    vx, vy = velocity.chunk(2, 1)
    t = expand_time(tau, x)
    return y-t*vy, x+(1-t)*vx  # earlier, later


def integrate(velocity_fn, initial, context, *, steps=20, method="heun", direction=1):
    if steps < 1 or method not in {"euler", "heun"} or direction not in (-1,1):
        raise ValueError("Invalid ODE solver specification")
    z = initial
    grid = torch.linspace(0, 1, steps+1, device=z.device)
    if direction == -1:
        grid = grid.flip(0)
    for i in range(steps):
        t, tn = grid[i], grid[i+1]
        dt = tn-t
        v = velocity_fn(z, t.expand(len(z)), context)
        if method == "heun":
            vn = velocity_fn(z+dt*v, tn.expand(len(z)), context)
            z = z + dt*(v+vn)*.5
        else:
            z = z + dt*v
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite ODE trajectory")
    return z

