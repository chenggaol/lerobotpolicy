import torch
import torch.nn as nn

from .configuration_mine import MineConfig
from transformers import AutoModel
import torch.nn.functional as F

# ---------- 1. ENCODER: observations -> tokens ----------
class Encoder(nn.Module):
    def __init__(self, config: MineConfig):
        super().__init__()

        #dinov3 upload model state dict from hugging face
        self.dino = AutoModel.from_pretrained(config.dino_model)
        self.n_skip = 1 + self.dino.config.num_register_tokens #skips first 4 non patch tokens
        dino_dim = self.dino.config.hidden_size    
        self.freeze = config.freeze_encoder
        d = config.hidden_dim
        if config.freeze_encoder:
            self.dino.requires_grad_(False)
            self.dino.eval()

        #Need to normalize images using these because dino images trained this way
        self.register_buffer("img_mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))    
        #position encoder mlp
        #input shape 
        state_dim = config.robot_state_feature.shape[0]    # 6 joints
        self.pose_mlp = nn.Sequential(
            nn.Linear(state_dim, 4 * d), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * d, d),
        )
        
        #2x2 max pool: 14x14 patch grid -> 7x7 = 49 tokens per camera
        self.pool = nn.MaxPool2d(kernel_size=2)
        #project dino features to the transformer width
        self.img_proj = nn.Linear(dino_dim, d)

    def encode_images(self,x:torch.Tensor):
        #x: (M,3,H,W) floats in [0,1] => (M,49,dino_dim)
        #M = number of images, p = number of patches per image, feature size of each patch toekn
        
        #resize image to 224x224, dino train images were this size
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        #normalize using mean and std used on dino training images
        x= (x-self.img_mean)/self.img_std
        
        with torch.set_grad_enabled(not self.freeze):
            patches = self.dino(pixel_values=x).last_hidden_state[:,self.n_skip:] #(M,196,C)
        M,P,C = patches.shape
        side = int(P**0.5) #side = 14
        grid = patches.transpose(1,2).reshape(M,C,side,side) #(M,C,14,14)
        grid = self.pool(grid)
        return grid.flatten(2).transpose(1,2) # (M,49,C)

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # images: one (B, T, 3, H, W) tensor per camera   state: (B, T, state_dim)
        # return: (B, T, 99, hidden_dim)   49 front + 49 wrist + 1 pose
        #feed into dinov3 model
        if state.dim() == 2: #means n_obs_steps = 1
            state = state.unsqueeze(1)
            images = [img.unsqueeze(1) for img in images]
        
        B,T = state.shape[:2]
        n_cam = len(images)

        #combine the cameras and batch and timesteps all together so dino runs once, then reshape
        x = torch.stack(images,dim=2).flatten(0,2)  #n_cam dim created from stack() (B*T*n_cam, 3, H, W)
        img_token = self.img_proj(self.encode_images(x)) # (M,49,C)
        img_token = img_token.reshape(B,T,-1,img_token.shape[-1])#(B,T,49*num_cam,C)

        pose_token = self.pose_mlp(state).unsqueeze(2) # (B,T,1,d)
        #combine image and pose token
        return torch.cat([img_token,pose_token],dim=2) # (B,T,49*2 + 1 = 99,d)
         


# ---------- 2. POLICY HEAD: tokens -> action chunk ----------
class ActionTransformer(nn.Module):
    def __init__(self, config: MineConfig):
        super().__init__()

        #hidden dim
        d, T, N = config.hidden_dim, config.n_obs_steps, config.n_tokens_per_step
        n_heads, dropout = config.n_heads, config.dropout
        action_dim = config.action_feature.shape[0]

        #learned summary token prepended to every timestep, filled in by spatial attention
        self.summary = nn.Parameter(torch.zeros(1, 1, 1, d))
        #position embeddings: which token within a timestep (+1 for summary), which timestep
        self.pos_space = nn.Parameter(torch.zeros(1, 1, N + 1, d))
        self.pos_time = nn.Parameter(torch.zeros(1, T, 1, d))

        def encoder_layer():
            return nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=4 * d,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        #token level/spatial level attention blocks (one per layer)
        self.spatial = nn.ModuleList(encoder_layer() for _ in range(config.n_layers))
        #temporal/timestep level attention blocks (one per layer)
        self.temporal = nn.ModuleList(encoder_layer() for _ in range(config.n_layers))
        #final norm, needed because norm_first layers leave their output un-normalized
        self.norm = nn.LayerNorm(d)

        #act decoder: one learned query per future action, cross-attends to all obs tokens
        self.action_queries = nn.Parameter(torch.zeros(1, config.horizon, d))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        #project each decoded query to one action
        self.action_out = nn.Linear(d, action_dim)

        for p in (self.summary, self.pos_space, self.pos_time, self.action_queries):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, tokens: torch.Tensor, obs_pad: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (B, n_obs, 25, hidden_dim)
        # obs_pad: (B, n_obs) bool, True = padded frame from before the episode started
        # return: (B, horizon, action_dim)
        #STANDARD TRANFORMER INPUT: (batch_size, sequence_length, d_model) when batch_first=True
        B, T, N, D = tokens.shape

        #prepend summary token to each timestep -> (B, T, N+1, D), then add positions
        x = torch.cat([self.summary.expand(B, T, 1, D), tokens], dim=2)
        x = x + self.pos_space + self.pos_time
        N = N + 1

        for spatial, temporal in zip(self.spatial, self.temporal):
            #fold time into batch so attention only mixes tokens within a timestep
            x = spatial(x.reshape(B * T, N, D)).reshape(B, T, N, D)
            #takes first token from spatial output as a summary token -> (B, T, D)
            summary = temporal(x[:, :, 0, :], src_key_padding_mask=obs_pad)
            #write updated summaries back so the next spatial layer sees temporal context
            x = torch.cat([summary.unsqueeze(2), x[:, :, 1:, :]], dim=2)

        #flatten every token of every timestep into one sequence for the decoder
        memory = self.norm(x).reshape(B, T * N, D)
        memory_pad = obs_pad.repeat_interleave(N, dim=1) if obs_pad is not None else None

        queries = self.action_queries.expand(B, -1, -1)                  # (B, horizon, D)
        out = self.decoder(tgt=queries, memory=memory, memory_key_padding_mask=memory_pad)
        return self.action_out(out)                                      # (B, horizon, action_dim)



# ---------- 3. CONNECTOR ----------
class MineModel(nn.Module):
    def __init__(self, config: MineConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.head = ActionTransformer(config)

    def forward(self, images, state, obs_pad=None):
        return self.head(self.encoder(images, state), obs_pad)