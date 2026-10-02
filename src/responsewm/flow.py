"""Coupled two-endpoint SymmFlow. Clinical time never equals flow time."""
from __future__ import annotations
import torch


def expand_time(t, ref):
    return t.reshape(len(t), *([1]*(ref.ndim-1)))


def make_joint_path(earlier_z, later_z, earlier_s, later_s, tau=None, generator=None):
    if earlier_z.shape != later_z.shape or earlier_s.shape != later_s.shape:
        raise ValueError("Endpoint shapes differ")
    b = len(earlier_z)
    tau = torch.rand(b, device=earlier_z.device, generator=generator) if tau is None else tau
    if tau.shape != (b,) or not torch.isfinite(tau).all() or ((tau < 0)|(tau > 1)).any():
        raise ValueError("Flow time must be [B] in [0,1]")
    def path(earlier, later):
        t = expand_time(tau, earlier)
        ex = torch.randn(later.shape, device=later.device, dtype=later.dtype, generator=generator)
        ey = torch.randn(earlier.shape, device=earlier.device, dtype=earlier.dtype, generator=generator)
        x = (1-t)*ex + t*later
        y = (1-t)*earlier + t*ey
        return torch.cat((x,y),1), torch.cat((later-ex,ey-earlier),1)
    z, vz = path(earlier_z,later_z)
    s, vs = path(earlier_s,later_s)
    return z,s,vz,vs,tau


def integrate(velocity, image, state, context, steps=20, method="heun", direction=1):
    if steps < 1 or method not in {"euler","heun"} or direction not in (-1,1):
        raise ValueError("Invalid ODE specification")
    # FP32 accumulator even when network evaluations are autocast to BF16.
    z, s = image.float(), state.float()
    grid = torch.linspace(0,1,steps+1,device=z.device)
    if direction < 0:
        grid = grid.flip(0)
    for t, tn in zip(grid[:-1],grid[1:]):
        dt = tn-t
        vz, vs, _ = velocity(z,s,t.expand(len(z)),context)
        vz, vs = vz.float(), vs.float()
        if method == "heun":
            vz2, vs2, _ = velocity(z+dt*vz,s+dt*vs,tn.expand(len(z)),context)
            vz, vs = (vz+vz2.float())*.5, (vs+vs2.float())*.5
        z, s = z+dt*vz, s+dt*vs
    if not torch.isfinite(z).all() or not torch.isfinite(s).all():
        raise FloatingPointError("Nonfinite ODE trajectory; do not silently clamp medical latents")
    return z,s
