# modeling_mine.py
import torch
import torch.nn as nn
from typing import Any

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from .configuration_mine import MineConfig

from .custom import MineModel

import torch.nn.functional as F
from lerobot.utils.constants import ACTION, OBS_STATE

from collections import deque

class MinePolicy(PreTrainedPolicy):
    config_class = MineConfig  # must match the string in @register_subclass
    name = "mine"

    def __init__(self, config: MineConfig, dataset_stats: dict[str, Any] = None, **kwargs):
        super().__init__(config, dataset_stats)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        self.model = MineModel(config)
        self.reset()


    def reset(self):
        """Reset per-episode state. Called by lerobot-eval at the start of each episode."""
        #last n_obs_steps observations, filled one per step by select_action
        self._obs_queue = deque(maxlen=self.config.n_obs_steps)
        #predicted actions waiting to be executed
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        #only trainable params: frozen DINO weights are left out of the optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        #batch already holds the full observation window: images (B, T, 3, H, W), state (B, T, 6)
        self.eval()
        images = [batch[k] for k in self.config.image_features]
        return self.model(images, batch[OBS_STATE], batch.get(f"{OBS_STATE}_is_pad"))

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called every step at inference)."""
        #batch holds one frame: images (B, 3, H, W), state (B, state_dim)
        keys = [OBS_STATE, *self.config.image_features]
        frame = {k: batch[k] for k in keys}
        if len(self._obs_queue) == 0:
            #episode start: pad the window with copies of the first frame, like the dataset loader does
            for _ in range(self.config.n_obs_steps - 1):
                self._obs_queue.append((frame, True))
        self._obs_queue.append((frame, False))

        if len(self._action_queue) == 0:
            #stack the window along a new time dim -> (B, T, ...)
            window = {k: torch.stack([f[k] for f, _ in self._obs_queue], dim=1) for k in keys}
            pad = torch.tensor([p for _, p in self._obs_queue], device=window[OBS_STATE].device)
            window[f"{OBS_STATE}_is_pad"] = pad.unsqueeze(0).expand(window[OBS_STATE].shape[0], -1)
            chunk = self.predict_action_chunk(window)[:, : self.config.n_action_steps]  # (B, n_action_steps, action_dim)
            #queue holds one (B, action_dim) action per step
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Compute the training loss.

        Returns `(loss, output_dict)`. `output_dict` may be `None`; everything in it must be
        logging-friendly Python natives (no tensors with gradients).

        `batch["action_is_pad"]` is a bool mask of shape (B, horizon) that marks
        timesteps padded because the episode ended before `horizon` steps; you
        can exclude those from your loss.
        """
        #one (B, T, 3, H, W) tensor per camera, same order every time
        images = [batch[k] for k in self.config.image_features]
        #image features is a dict of ju camera features(observation.images.front and wrist)
        #(B, T) True = frame from before the episode started
        obs_pad = batch.get(f"{OBS_STATE}_is_pad")

        #forward pass
        pred = self.model(images, batch[OBS_STATE], obs_pad) # (B, horizon, action_dim)
        actions = batch[ACTION]

        #loss (shape (batch,action_horizon,6(joints+gripper state)))
        loss = F.l1_loss(pred, actions, reduction="none")

        #increase gripper position weighting
        weights = torch.ones(loss.shape[-1], device = loss.device)
        weights[-1] = 4.0 #gripper loss set as four times seperate joint losses
        loss = loss * weights
        
        #ignore action steps past the end of the episode
        action_is_pad = batch.get("action_is_pad") #(B, horizon)
        if action_is_pad is not None:
            mask = ~action_is_pad.unsqueeze(-1) # (B, horizon, 1)
            loss = (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp(min=1)
        else:
            loss = loss.mean()


        return loss, {"l1_loss": loss.item()}