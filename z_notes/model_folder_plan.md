# Plan: a lerobot-compatible `my_policy/` for the configurable encoder + head model

Target from `model_plan.txt`: 1-2 cameras + joint state + past-state window in;
configurable image encoder (DINO / custom CNN) emitting a configurable number of
tokens; configurable main policy (ACT-style transformer / MLP); action chunk out
with configurable horizon.

---

## 0. Your question first: can the encoder/head variants share the same files?

**Yes — and with lerobot you basically have to.** One registered policy type,
variants selected by config strings.

The reason is the resolution convention in `policies/factory.py:389-435`. When
you register a config under a name, lerobot derives everything else by string
rewriting:

| Registered name | `"mine"` |
|---|---|
| Config class | `MineConfig` — **must** end in `Config` |
| Config module | `.../configuration_mine.py` |
| Policy class | `MinePolicy` in `.../modeling_mine.py` — `configuration_` → `modeling_` |
| Processor fn | `make_mine_pre_post_processors` in `.../processor_mine.py` |

So each registered type gets exactly one `modeling_*` module. If you registered a
type per combination you'd need `mine_dino_act`, `mine_dino_mlp`, `mine_cnn_act`,
`mine_cnn_mlp` — four registrations, four config classes, and the same four again
every time you add a backbone. Wrong axis to multiply on.

Instead: **`image_encoder` and `head` are `str` fields on one `MineConfig`**, and
`modeling_mine.py` holds a registry dict per axis.

```python
IMAGE_ENCODERS = {"dino": DinoEncoder, "cnn": CnnEncoder}
HEADS          = {"act": ACTHead,      "mlp": MLPHead}
```

The property that makes this actually work: those strings are dataclass fields, so
they round-trip through `config.json` via `_save_pretrained`. `from_pretrained`
reads the checkpoint's config, sees `image_encoder="dino"`, and rebuilds the right
architecture with no extra flags at load time. A checkpoint is self-describing.

The convention only pins those *three* module names. Extra sibling modules are
free, so split by axis once the file gets long:

```
my_policy/
├── __init__.py              # import config + modeling so @register_subclass fires
├── configuration_mine.py    # MineConfig
├── modeling_mine.py         # MinePolicy + MineModel + the two registry dicts
├── encoders.py              # DinoEncoder, CnnEncoder, StateEncoder, PastActionEncoder, TokenResampler
├── heads.py                 # ACTHead, MLPHead
└── processor_mine.py        # make_mine_pre_post_processors
```

Keep `modeling_mine.py` as assembly + the `PreTrainedPolicy` surface only. Put the
architectures in `encoders.py` / `heads.py`. `get_policy_class` never looks at
them; it only needs `MinePolicy` to be importable from `modeling_mine`.

---

## 1. The one design decision everything else hangs off: the token contract

Fix a single interface and the 2×2 stops being four code paths:

> **Every encoder returns `(B, n_tokens, dim_model)`. Every head consumes
> `(B, total_tokens, dim_model)` and returns `(B, chunk_size, action_dim)`.**

- image encoder → `(B, n_image_tokens, dim_model)` — configurable count
- state encoder → `(B, 1, dim_model)` — `nn.Linear(state_dim * n_obs_steps, dim_model)`
- past-action encoder → `(B, 1, dim_model)` — same shape, optional
- `MineModel.forward` concatenates along dim 1, adds a learned per-modality
  embedding, and hands the sequence to whichever head is configured

The ACT head attends over the sequence. The MLP head flattens it
(`total_tokens * dim_model`) and runs an MLP. Both see identical input, so encoder
choice and head choice are genuinely independent, and an ablation is a config edit.

### Making `n_image_tokens` actually configurable

Neither backbone naturally produces a chosen number:

- ResNet18 at 480×640 → a 15×20 feature map → 300 tokens, set by resolution, not by you
- DINOv2 (patch 14) at 224² → 256 patch tokens + 1 CLS

ACT just flattens the CNN grid and lives with whatever falls out
(`modeling_act.py:475-481`). Two ways to make it a knob:

1. **Adaptive pool** the feature map to `g×g` before projecting. One line, free,
   but token count is restricted to perfect squares and pooling throws away
   spatial detail bluntly.
2. **Learned resampler** (recommended): `n_image_tokens` learned query embeddings
   cross-attend once over the backbone's patch/cell tokens. Any count, one
   `nn.MultiheadAttention`, and it is identical code for DINO and CNN — the only
   thing that varies upstream is the token dim, which a `1×1` conv or `nn.Linear`
   projection normalizes to `dim_model`.

Put the resampler in the shared path, *after* whichever backbone. Then the backbone
axis and the token-count axis are orthogonal, which is the point.

**Per-camera or shared?** With 2 cameras, `n_image_tokens` should mean *per camera*
(so 2 cams × 8 tokens = 16 image tokens) and the cameras should share one backbone's
weights but get a distinct learned camera embedding added to their tokens. Shared
weights halve the parameter count and are standard; the embedding is what keeps
"wrist" and "front" distinguishable.

---

## 2. Config surface

```python
@PreTrainedConfig.register_subclass("mine")
@dataclass
class MineConfig(PreTrainedConfig):
    # --- temporal shape ---
    n_obs_steps: int = 1          # past-state window (1 = current only)
    chunk_size: int = 16          # predicted horizon
    n_action_steps: int = 8       # executed before re-planning; must be <= chunk_size
    n_past_actions: int = 0       # past-action window (0 = feature off)

    # --- axis 1: image encoder ---
    image_encoder: str = "cnn"    # "cnn" | "dino"
    cnn_backbone: str = "resnet18"
    dino_model: str = "facebook/dinov2-small"
    pretrained_backbone: bool = True
    freeze_image_encoder: bool = False
    n_image_tokens: int = 8       # per camera, via the resampler
    share_camera_encoder: bool = True

    # --- axis 2: head ---
    head: str = "act"             # "act" | "mlp"
    dim_model: int = 256          # the token width; shared by everything
    # act:
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1     # ACT's own decoder is 1 layer; more rarely helps
    n_heads: int = 8
    dim_feedforward: int = 1024
    # mlp:
    mlp_hidden: int = 512
    mlp_layers: int = 3

    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5   # backbones want a smaller lr than the head
```

Required overrides (all abstract on `PreTrainedConfig`, `configs/policies.py:106-131`):

```python
@property
def observation_delta_indices(self):  return list(range(1 - self.n_obs_steps, 1))
@property
def action_delta_indices(self):       return list(range(-self.n_past_actions, self.chunk_size))
@property
def reward_delta_indices(self):       return None

def get_optimizer_preset(self):  return AdamWConfig(lr=self.optimizer_lr)
def get_scheduler_preset(self):  return None

def validate_features(self):
    if not self.image_features:
        raise ValueError("need at least one camera")
    if self.n_action_steps > self.chunk_size:
        raise ValueError("n_action_steps must be <= chunk_size")
    if self.image_encoder not in IMAGE_ENCODERS: ...
    if self.head not in HEADS: ...
```

Validating the two strings in `validate_features` matters: it turns a typo into an
error at construction with the valid choices listed, instead of a `KeyError` deep
in `__init__`.

---

## 3. How each of the four inputs is actually plumbed

This is where the time goes. Train and inference deliver the same tensors by
**completely different mechanisms**, and they have to agree exactly.

| Input | Train (from dataset) | Inference (from robot) |
|---|---|---|
| images | `(B, n_obs_steps, C, H, W)` per cam key, via `observation_delta_indices` | policy's own deque, stacked `dim=1` |
| state | `(B, n_obs_steps, 6)`, same | same deque mechanism |
| past actions | `(B, n_past_actions + chunk_size, 6)` — negative `action_delta_indices` | **not in the observation at all** — see below |
| action target | the `chunk_size` positive slice of that same tensor | n/a |

**Observation history.** `observation_delta_indices` makes the *dataset* stack the
window; the robot has no such thing. Copy the diffusion policy's approach exactly
(`modeling_diffusion.py:94-113`): allocate `deque(maxlen=n_obs_steps)` per key in
`reset()`, fill with `populate_queues` from `lerobot.policies.utils`, then
`torch.stack(list(q), dim=1)`. `populate_queues` back-fills a short queue by
repeating the first observation, so step 0 of an episode is handled for you.

For the backbone, fold time into batch: `(B, T, C, H, W)` → `(B*T, C, H, W)` →
encode → unfold → concatenate along the token axis or the feature axis. Folding is
correct and simple; a temporal transformer over frames is a later upgrade.

Normalization is safe with the extra time axis — visual stats are reshaped to
`(C,1,1)` (`normalize_processor.py:139-155`) and state stats stay `(D,)`, so both
broadcast against a `(B, T, ...)` tensor unchanged. No processor work needed.

**Past actions — read this before building it.** There is no first-class past-action
observation feature anywhere in lerobot; I checked the whole `policies/` tree. So:

- *Train*: request negative `action_delta_indices` (diffusion already does this —
  `configuration_diffusion.py:256`), then slice the returned tensor into
  `past = a[:, :n_past_actions]` and `target = a[:, n_past_actions:]`.
- *Inference*: `action` is not in the batch the robot hands you. `MinePolicy` must
  push every action it emits into its own `deque(maxlen=n_past_actions)` inside
  `select_action`, and inject it into the batch itself.

That asymmetry is a genuine train/inference distribution shift: at train time the
model conditions on *ground-truth* past actions, at run time on *its own* past
outputs, and errors compound. It is the classic behavioral-cloning failure mode, and
it can make a model that trains beautifully drift on hardware.

**My recommendation: build the plumbing, default `n_past_actions = 0`.** On the
SO-101 the action space is joint *position targets* and `observation.state` is
measured joint positions, so a past action is very nearly a past state shifted by
one tick — you are already getting that signal from `n_obs_steps`. Keep it as an
ablation you can switch on, not a thing you debug on day one. If you do turn it on,
the standard mitigations are small `n_past_actions`, additive noise on the past
actions during training, or scheduled sampling.

**Bootstrap at episode start** (no past actions exist yet): pad with the current
state, or with zeros — but do it *identically* in the train and inference paths, or
you get a silent mismatch on the first `n_past_actions` steps of every episode.

**Chunking.** `chunk_size` is what the model predicts; `n_action_steps` is what gets
executed before re-planning; queue the difference. `select_action` pops from a
`deque(maxlen=n_action_steps)` and only calls the network when it's empty, so
inference cost is amortized to once every `n_action_steps` ticks. `reset()` must
clear both the action queue and the observation deques or a stale action fires first
on the next episode.

---

## 4. Dependency note

`transformers` and `timm` are **not installed** in `.venv` (torch 2.11.0+cu130,
torchvision 0.26.0 are). DINOv2 needs one of them. So:

- `cnn` path works today on torchvision alone
- `dino` path needs `uv pip install transformers` first

Import the DINO backbone *lazily inside `DinoEncoder.__init__`*, not at module
scope. `get_policy_class` imports `modeling_mine` on every load, so a top-level
`import transformers` would break the CNN path for anyone without it — which is
exactly why lerobot's own factory comments call out keeping heavy optional deps
lazy (`factory.py:85-88`).

---

## 5. Build order

Each phase ends with something you can actually check, so a break is localized.

1. **Skeleton that loads.** `MineConfig` + `MinePolicy` with `image_encoder="cnn"`,
   `head="mlp"`, `n_obs_steps=1`, `chunk_size=1`, no past actions. Essentially your
   existing `policy_structure/my_policy/` scaffold, which is already close.
   *Check:* `MinePolicy(cfg)` constructs, `save_pretrained` then `from_pretrained`
   round-trips, `forward` on a fake batch returns a scalar loss.

2. **Token contract.** Refactor to `(B, n_tokens, dim_model)` everywhere; add the
   resampler; `n_image_tokens` becomes real. Still CNN + MLP.
   *Check:* changing `n_image_tokens` changes the token count and nothing crashes;
   loss still decreases on a tiny overfit batch.

3. **Second axis.** Add `ACTHead` and the `HEADS` registry.
   *Check:* all four (cnn|dino not yet) — i.e. cnn×mlp and cnn×act — construct and
   overfit 10 samples to near-zero loss. Overfitting a handful of samples is the
   fastest real test that the wiring is right.

4. **Chunking + history.** `chunk_size > 1`, `n_action_steps > 1`, `n_obs_steps > 1`,
   with the deque machinery in `select_action`.
   *Check:* the action queue empties and refills every `n_action_steps`; the batch
   the model sees at inference has the same shape as the one it saw at train time.
   Assert this explicitly — it is the single most common silent bug here.

5. **DINO encoder.** Lazy import, frozen by default, projection to `dim_model`.
   *Check:* both encoders produce the same token shape.

6. **Second camera.** Shared weights + per-camera embedding.
   *Check:* token count scales as `n_cams * n_image_tokens`.

7. **Past actions**, only if you want them, per the warning above.

Phases 1-3 are worth doing before you have much data; they are pure shape-plumbing
and you can test them with random tensors.

---

## 6. Two things to fix in your existing notes

- `policy_structure.txt` describes the layout accurately — but `scripts/run_arm.py`
  imports `armcurl_policy`, which is now under `archived/`, and `poses.py` has moved
  to `scripts/`. Those imports are broken right now.
- The *Python import path* is `lerobot.robots.so_follower` (the `so101_follower`
  module no longer exists), but the **CLI type name is still `so101_follower`** —
  `config_so_follower.py:56` registers both `so101_follower` and `so100_follower`.
  So the `lerobot-rollout` lines in `inference_notes.txt` are correct as written;
  only Python `import` statements need updating.

---

## 7. Open decisions for you

1. **`dim_model = 256` for both heads?** Shared width is what makes the token
   contract clean. Only reason to split is if the MLP head wants to be much smaller.
2. **Is the MLP head meant to be a real baseline or a debugging shim?** If it's a
   baseline, it should get the same chunk output and be trained the same way. If it's
   a shim to check plumbing without transformer cost, it can stay minimal.
3. **VAE objective?** ACT's real gains come substantially from its CVAE latent, not
   the transformer alone. A plain transformer head labeled "act" will underperform
   published ACT numbers. Worth knowing before you compare.
4. **Loss.** L1 on the chunk is the ACT default and a fine start. Uniform weighting
   across the horizon is standard; discounting later steps is a knob if the far end
   of the chunk is noisy.
