# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlowMatchScheduler for Causal-Forcing few-step inference.

Faithful port of the flow-matching scheduler from thu-ml/Causal-Forcing
(``utils/scheduler.py``). The framewise inference loop never calls ``step`` —
it only needs:

- the ``sigmas`` / ``timesteps`` schedule (to warp the discrete denoising step
  list and to convert the model's flow prediction to x0), and
- ``add_noise`` (to re-noise the x0 prediction between steps of the multi-step
  first chunk).

The model predicts flow/velocity; x0 = x_t - sigma_t * flow_pred.
"""

from __future__ import annotations

import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int = 1000, denoising_strength: float = 1.0) -> None:
        # Pin the schedule to CPU: it is a small constant table moved per-op onto
        # the input device (_sigma_for / convert_flow_to_x0). Under the engine's
        # default-device context an unqualified linspace would land on the
        # accelerator and later clash with CPU tensors in warp_denoising_step.
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1, device="cpu")[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps, device="cpu")
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def _sigma_for(self, timestep: torch.Tensor) -> torch.Tensor:
        """Nearest sigma for each timestep value (double precision index match)."""
        timesteps = self.timesteps.to(timestep.device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        return self.sigmas.to(timestep.device)[timestep_id]

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward corruption: sample = (1 - sigma) * x0 + sigma * noise.

        Shapes follow the upstream flatten(0, 1) convention: original_samples /
        noise are [B*F, C, H, W] and timestep is [B*F].
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigma = self._sigma_for(timestep).reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def convert_flow_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """x0 = x_t - sigma_t * flow_pred (computed in double precision).

        flow_pred / xt are [B*F, C, H, W]; timestep is [B*F].
        """
        original_dtype = flow_pred.dtype
        flow_pred = flow_pred.double()
        xt = xt.double()
        sigma = self._sigma_for(timestep.double()).reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma * flow_pred
        return x0_pred.to(original_dtype)

    def warp_denoising_step(self, denoising_step_list: list[int]) -> torch.Tensor:
        """Map integer step indices to warped continuous timesteps.

        Matches upstream: ``timesteps = cat(scheduler.timesteps, [0])`` then index
        by ``1000 - denoising_step_list``.
        """
        # Build on CPU explicitly: under the engine's default-device context an
        # unqualified torch.tensor(...) would land on the accelerator and clash
        # with the CPU timesteps in torch.cat.
        steps = torch.tensor(denoising_step_list, dtype=torch.long, device="cpu")
        timesteps_cpu = self.timesteps.detach().to("cpu")
        timesteps = torch.cat((timesteps_cpu, torch.zeros(1, dtype=torch.float32, device="cpu")))
        return timesteps[self.num_train_timesteps - steps]
