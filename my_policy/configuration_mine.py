# configuration_mine.py
from dataclasses import dataclass, field
from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

@PreTrainedConfig.register_subclass("mine")
@dataclass
class MineConfig(PreTrainedConfig):
    """Configuration class for MinePolicy.

    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon
        n_action_steps: Number of action steps to execute
        hidden_dim: Hidden dimension for the policy network
        # Add your policy-specific parameters here
    """
    #MAKE THESE ADJUSTABLE FLAGS
    horizon: int = 50
    n_action_steps: int = 50
    hidden_dim: int = 256

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    #number features
    model_dim = 384

    #how many observations are kept in queue
    n_obs_steps: int = 4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,   # Encoder applies DINO's ImageNet norm itself
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    #DINOv3 image encoder parameters
    dino_model = "facebook/dinov3-vits16-pretrain-lvd1689m"
    freeze_encoder = True
    n_tokens_per_step = 99 # 49 front + 49 wrist + 1 pose
    #action transformer parameters
    n_layers = 4  # spatial+temporal block pairs
    n_decoder_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")

    def validate_features(self) -> None:
        """Validate input/output feature compatibility.

        Call this explicitly from your policy's __init__ — the base class does not.
        """
        if not self.image_features:
            raise ValueError("MinePolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("MinePolicy requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        """Return a LRSchedulerConfig from lerobot.optim, or None."""
        return None
    
    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Relative timestep offsets the dataset loader provides per observation.

        Return `None` for single-frame policies. For temporal policies that consume
        multiple past or future frames, return a list of offsets, e.g. `[-20, -10, 0, 10]` for
        3 past frames at stride 10 and 1 future frame at stride 10.
        """
    
        return list(range(-(self.n_obs_steps-1),1)) 
        #for indices [-(n-1), ..., 0]

    #tells loader which frame's actions to return with each training sample
    #ie if action_horizon = 5 , return [0,1,2,3,4] 
    @property
    def action_delta_indices(self) -> list[int]:
        """Relative timestep offsets for the action chunk the dataset loader returns."""
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None