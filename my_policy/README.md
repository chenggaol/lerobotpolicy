# MinePolicy

A custom [LeRobot](https://github.com/huggingface/lerobot) policy for the SO-101 arm. It takes a short window of past camera frames and joint positions and predicts a chunk of future joint targets. At deployment it runs **asynchronously**: three independent fixed-rate clocks handle observing, planning and executing.

```
 front cam ─┐                                                       ┌─► 50 future joint targets
 wrist cam ─┼─► Encoder (DINOv3 + pose MLP) ─► ActionTransformer ───┤   (horizon × 6)
 joints  ───┘   per-timestep tokens            spatial ⟷ temporal   └─► executed by async runner
                (B, T, 99, 256)                attention + ACT decoder
```

- **Registered name:** `mine` (`--policy.type=mine`)
- **Size:** 30.4M parameters, of which 8.8M are trainable. The 21.6M DINOv3 backbone is frozen by default.
- **Inputs:** `observation.images.front`, `observation.images.wrist` (480×640 RGB) and `observation.state` (6 joints). Each covers the last `n_obs_steps` frames.
- **Output:** `action` with shape `(horizon, 6)`, the absolute joint position targets for the next `horizon` frames.

---

## Contents

1. [Files](#1-files)
2. [Quick start](#2-quick-start)
3. [How LeRobot finds this policy](#3-how-lerobot-finds-this-policy)
4. [Configuration](#4-configuration)
5. [Data pipeline](#5-data-pipeline)
6. [Architecture](#6-architecture)
7. [Training objective](#7-training-objective)
8. [Asynchronous inference](#8-asynchronous-inference)
9. [Tuning guide](#9-tuning-guide)
10. [Known limitations and TODOs](#10-known-limitations-and-todos)

---

## 1. Files

| File | Contents |
|---|---|
| `configuration_mine.py` | `MineConfig`: every hyperparameter, normalization, and the frame offsets the dataset loader should return |
| `custom.py` | The network: `Encoder`, `ActionTransformer`, and `MineModel` (the two joined together) |
| `modeling_mine.py` | `MinePolicy`: the LeRobot wrapper (training loss, `predict_action_chunk`, optimizer parameters) |
| `processor_mine.py` | `make_mine_pre_post_processors`: normalization before the model and unnormalization after it |
| `async_runner.py` | Asynchronous deployment on the SO-101: three clocks plus a background planner thread |
| `__init__.py` | Empty, but required so that `my_policy` is an importable package |
| `plan.txt` | Original design notes |

---

## 2. Quick start

Run all commands from the repo root (`~/lerobot`) using the `.venv1` environment.

### One-time setup

```bash
# DINOv3 needs transformers>=4.56. Pin huggingface-hub: LeRobot 0.6 requires <2.0,
# and installing transformers on its own upgrades the hub past that.
.venv1/bin/pip install "transformers>=4.56" accelerate "huggingface-hub>=1.31,<2.0"
```

The DINOv3 weights are gated. Accept the license at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> using the same account you are logged in with (`hf auth login`).

### Train

```bash
PYTHONPATH=. .venv1/bin/lerobot-train \
    --policy.type=mine \
    --policy.discover_packages_path=my_policy \
    --policy.push_to_hub=false \
    --dataset.repo_id=chenggao/so101_pick_cube_20260921_200844 \
    --batch_size=8 --steps=50000 \
    --output_dir=outputs/train/mine_pick_cube
```

- `PYTHONPATH=.` is required. The `lerobot-train` console script doesn't put the current directory on the Python path, so without it the run fails with `No module named 'my_policy'`.
- Override any config field from the command line, for example `--policy.horizon=16 --policy.n_obs_steps=2`. This only works for fields that have a type annotation (see [§4](#4-configuration)).
- Checkpoints are written to `outputs/train/<run>/checkpoints/<step>/pretrained_model/`. Each one contains the weights, `config.json` and the saved normalizer statistics.

### Deploy on the SO-101 (asynchronous)

```bash
.venv1/bin/python -m my_policy.async_runner \
    --checkpoint outputs/train/mine_pick_cube/checkpoints/last/pretrained_model \
    --dataset-repo-id chenggao/so101_pick_cube_20260921_200844 \
    --port /dev/ttyACM0 --robot-id <follower calibration id> \
    --front-cam 0 --wrist-cam 2 \
    --replan-hz 5 --duration 60
```

The camera indices, port and robot id must match the setup used when the dataset was recorded.

---

## 3. How LeRobot finds this policy

LeRobot finds the policy by naming convention, so names in different files must agree with each other:

| LeRobot looks for | Rule | Here |
|---|---|---|
| Config class | registered via `@PreTrainedConfig.register_subclass("mine")` | `MineConfig` |
| Policy module | `configuration_X` → `modeling_X` | `modeling_mine` |
| Policy class | config class name with `Config` replaced by `Policy` | `MinePolicy` |
| Processor module | `configuration_X` → `processor_X` | `processor_mine` |
| Processor factory | `make_{registered name}_pre_post_processors` | `make_mine_pre_post_processors` |

If you rename the policy, rename all five together. A config class called `MyPolicyConfig`, for example, makes LeRobot look for a class called `MyPolicyPolicy`.

`--policy.discover_packages_path=my_policy` imports the package before the command-line arguments are parsed, which is what registers `"mine"`.

LeRobot calls `MinePolicy(config, dataset_stats=..., dataset_meta=...)`. The constructor takes `**kwargs` so that it accepts the extra arguments.

---

## 4. Configuration

All fields live in `MineConfig` (`configuration_mine.py`). Fields with a type annotation (`name: type = value`) are real dataclass fields: you can set them from the command line and they are saved in checkpoints. Fields without one are plain class constants (see [§10](#10-known-limitations-and-todos)).

### Time

| Field | Default | Meaning |
|---|---|---|
| `n_obs_steps` | `4` | Length of the observation window, in frames. At 30 fps, 4 frames cover about 0.1 s. |
| `horizon` | `50` | Number of future actions predicted per plan. At 30 fps, 50 actions cover about 1.67 s. |
| `n_action_steps` | `50` | Only used by the synchronous `select_action` path, which isn't implemented yet. The async runner executes actions by time instead. Must be ≤ `horizon`. |

### Encoder

| Field | Default | Meaning |
|---|---|---|
| `dino_model` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Hugging Face model id for the backbone. ViT-S/16 has 21M parameters and 384-dimensional features. |
| `freeze_encoder` | `True` | Freezes DINOv3. When frozen, it runs under `no_grad` and is left out of the optimizer. |
| `n_tokens_per_step` | `99` | Tokens per timestep: 49 per camera × 2 cameras + 1 pose token. This must match the encoder output (see [§10](#10-known-limitations-and-todos)). |

### Transformer

| Field | Default | Meaning |
|---|---|---|
| `hidden_dim` | `256` | Width of every token (`d_model`). Must be divisible by `n_heads`. |
| `n_layers` | `4` | Number of spatial + temporal attention block pairs. |
| `n_decoder_layers` | `2` | Number of action-decoder layers. |
| `n_heads` | `8` | Attention heads. With `hidden_dim=256` that gives 32 dimensions per head. |
| `dropout` | `0.1` | Dropout in the attention, MLP and pose-MLP layers. |

### Optimization

| Field | Default | Meaning |
|---|---|---|
| `optimizer_lr` | `1e-4` | AdamW learning rate |
| `optimizer_weight_decay` | `1e-4` | AdamW weight decay |
| scheduler | `None` | Constant learning rate (`get_scheduler_preset`) |

### Normalization (`normalization_mapping`)

| Feature type | Mode | Why |
|---|---|---|
| `VISUAL` | `IDENTITY` | Images go into the model in [0, 1]. The encoder applies DINOv3's own ImageNet mean and std, so normalizing them with dataset statistics as well would normalize them twice. |
| `STATE` | `MEAN_STD` | Puts every joint on the same scale before the pose MLP. |
| `ACTION` | `MEAN_STD` | The loss is computed in normalized units so every joint counts equally. Actions are unnormalized after the model, before they reach the motors. |

### Derived from the dataset (not set by you)

LeRobot fills in `input_features` and `output_features` from the dataset metadata. The base class then provides:

- `config.image_features`: `{"observation.images.front": …, "observation.images.wrist": …}`
- `config.robot_state_feature`: `observation.state`, with shape `(6,)`
- `config.action_feature`: `action`, with shape `(6,)`

The model reads its shapes from these, so it doesn't hardcode the camera names or the number of joints.

---

## 5. Data pipeline

### Dataset

The training data is the SO-101 pick-cube recordings in `~/.cache/huggingface/lerobot/chenggao/`:

| Key | Shape (one frame) | Contents |
|---|---|---|
| `observation.state` | `(6,)` | `.pos` of `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` and `gripper` |
| `observation.images.front` | `(480, 640, 3)` | Front camera |
| `observation.images.wrist` | `(480, 640, 3)` | Wrist camera |
| `action` | `(6,)` | Joint position targets, with the same names as the state |

### Windows: `*_delta_indices`

The config tells the dataset loader which frames, relative to the sampled frame `t`, it should return:

```python
observation_delta_indices = range(-(n_obs_steps - 1), 1)   # [-3, -2, -1, 0] → past frames
action_delta_indices      = range(horizon)                  # [0, 1, …, 49]   → future actions
```

LeRobot converts these offsets to timestamps (`index / fps`) and stacks the frames, so a training batch looks like this:

```
observation.images.front   (B, 4, 3, 480, 640)
observation.images.wrist   (B, 4, 3, 480, 640)
observation.state          (B, 4, 6)
observation.state_is_pad   (B, 4)       True = frame from before the episode started
action                     (B, 50, 6)
action_is_pad              (B, 50)      True = step past the end of the episode
```

### Padding masks

The loader always returns full-size windows, and fills frames that don't exist with copies of the nearest real one:

```
start of episode (t = 1):          end of episode (200 frames, t = 180):
 obs frames  -2  -1   0   1         actions 180 … 199 | 200 … 229
 given       f0  f0  f0  f1                 real (20) | copies of a199 (30)
 is_pad       T   T   F   F                 False     | True
```

- `observation.state_is_pad` goes into the **model** as `obs_pad`. Attention ignores the padded frames (see [§6.3](#63-masking)).
- `action_is_pad` goes into the **loss**, so the model isn't graded on actions past the end of an episode (see [§7](#7-training-objective)).

### Processors

`make_mine_pre_post_processors` uses LeRobot's default pipelines:

- **Preprocessor:** rename keys → add a batch dimension → move to `config.device` → normalize (`normalization_mapping` above)
- **Postprocessor:** unnormalize the action → move it to the CPU

The normalizer statistics are saved with every checkpoint, so deployment uses exactly the statistics from training.

---

## 6. Architecture

All tensors follow the layout `(batch B, timestep T, token N, feature D)`.

### 6.1 Encoder: observations → tokens (`custom.py: Encoder`)

```
images: list of (B, T, 3, 480, 640), one per camera        state: (B, T, 6)
   │ stack cameras, fold B·T·cams into one batch dim               │
   ▼                                                               │
(B·T·2, 3, 480, 640)                                               │
   │ resize 224×224 (bilinear, antialias)                          │
   │ ImageNet mean/std                                             │
   ▼                                                               │
DINOv3 ViT-S/16 (frozen)                                           │
   │ last_hidden_state: [CLS, 4 registers, 196 patches]            │
   │ drop CLS + registers → (B·T·2, 196, 384)                      │
   │ reshape to 14×14 grid → 2×2 max pool → 7×7 = 49 tokens        │
   │ Linear 384 → 256                                              ▼
   ▼                                                    pose MLP: Linear 6→1024, GELU,
(B, T, 98, 256)  [front 0–48 | wrist 49–97]             Dropout, Linear 1024→256
   └──────────────────── concat ───────────────────────────┐   (B, T, 1, 256)
                                                           ▼
                                             tokens (B, T, 99, 256)
```

- **One DINOv3 call.** Batch, time and camera are folded into a single batch dimension, so DINOv3 runs once per forward pass rather than in a loop.
- **Why max pooling to 7×7?** It's a clean 2×2 pool of the 14×14 grid, with no uneven windows. Each token covers about 32×32 px of the 224 image, which is enough detail for gripper–cube alignment. Max pooling keeps strong responses from small objects, which average pooling would wash out. It also has no parameters, which matters with small datasets. A learned query resampler (Perceiver-style) is a drop-in alternative if more capacity is needed.
- **Aspect ratio.** The 4:3 frames are squashed to a square. Every frame is squashed the same way, so the policy can learn from them anyway. Resizing to 224×288 instead (a 14×18 grid → 7×9 = 63 tokens) keeps the aspect ratio, but `n_tokens_per_step` and the grid reshape would have to change.
- **Pose token.** The MLP maps the 6 joint values at each timestep to one 256-dimensional token. There is no activation after the last layer, so the token can take any value, like the image tokens.
- **Single frame.** If `state` arrives as `(B, 6)` with no T dimension, a T of 1 is added, so the head always receives 4-D input.

### 6.2 ActionTransformer: tokens → action chunk (`custom.py: ActionTransformer`)

```
tokens (B, T, 99, D)
  │ prepend a learned summary token per timestep       → (B, T, 100, D)
  │ + pos_space (1, 1, 100, D)   which token: summary / front patch / wrist patch / pose
  │ + pos_time  (1, T, 1, D)     which timestep
  ▼
┌──────────── × n_layers ─────────────────────────────────────────────────┐
│ spatial:  TransformerEncoderLayer on (B·T, 100, D)                      │
│           tokens attend within their own timestep                       │
│ temporal: TransformerEncoderLayer on the summary tokens only (B, T, D)  │
│           summaries attend across timesteps (key padding mask = obs_pad)│
│ write the updated summaries back into slot 0                            │
└─────────────────────────────────────────────────────────────────────────┘
  │ LayerNorm, flatten every token of every timestep  → memory (B, T·100, D)
  ▼
ACT decoder (TransformerDecoder × n_decoder_layers)
  queries: `horizon` learned action queries (1, horizon, D), shared by all samples
    self-attention:  queries attend to each other (keeps the chunk consistent)
    cross-attention: Q = queries, K = V = memory   (memory_key_padding_mask)
    MLP
  ▼
Linear D → 6                                        → actions (B, horizon, 6)
```

All attention layers use pre-norm (`norm_first=True`), GELU, `dim_feedforward = 4·D`, and `batch_first=True`.

**Why factorized attention?** Plain attention over every token is `(T·100)²`. Splitting it into spatial attention (`100²` per timestep) and temporal attention (`T²` over the summaries) is much cheaper and reflects the structure of the data: the scene at one moment, then how it changes over time. This is the same layout as TimeSformer and ViViT.

**How the summary token works.** It starts as a learned vector identical for every timestep. During spatial attention it collects information from its timestep's 99 tokens, much like `[CLS]` in ViT, and it is the only token temporal attention looks at. Because the updated summary is written back, the next block's spatial attention can pass temporal context on to the image and pose tokens. So stacking blocks alternates between "what's in this frame" and "how frames relate".

**Why does `x[:, :, 0]` represent a whole timestep?** Indexing with an integer removes that dimension: `(B, T, N, D)[:, :, 0]` becomes `(B, T, D)`, which is one token per timestep. `unsqueeze(2)` adds the dimension back before concatenating.

**No causal mask.** Every frame in the window is already in the past when the policy acts, so letting older frames attend to newer ones leaks nothing. The order comes from `pos_time`. The decoder's action queries are not causally masked either: all `horizon` actions are predicted in parallel, as in ACT.

**Decoder Q/K/V.** In cross-attention, the queries come from the action queries (which future action), and the keys and values come from the observation tokens (what was seen). The output length follows Q (`horizon`), and the content comes from K/V.

### 6.3 Masking

`obs_pad` (B, T), which is `observation.state_is_pad`:

- **Temporal attention:** `src_key_padding_mask=obs_pad`. Summary tokens can't attend to padded timesteps.
- **Decoder cross-attention:** `obs_pad.repeat_interleave(100, dim=1)` expands it to one flag per memory token, so action queries can't read from padded timesteps.
- **Spatial attention** needs no mask, because it only mixes tokens within one timestep, and nothing downstream reads from a padded timestep.

The current frame is never padded, so attention always has at least one real frame.

---

## 7. Training objective

`MinePolicy.forward` computes the mean squared error (L2) between the predicted chunk and the ground truth, averaged over real steps only:

```python
pred  = model(images, state, obs_pad)                    # (B, horizon, 6), normalized
loss  = F.mse_loss(pred, batch["action"], reduction="none")
mask  = ~batch["action_is_pad"].unsqueeze(-1)            # (B, horizon, 1)
loss  = (loss * mask).sum() / (mask.sum() * 6).clamp(min=1)
```

- `reduction="none"` keeps the per-element errors so padded steps can be zeroed out.
- Dividing by the number of real elements, rather than using `.mean()`, keeps the loss on the same scale whether a sample has 0 or 40 padded steps.
- Without the mask, the model would learn "hold the final pose" for every step past the end of an episode, and would stop early near the end of each demonstration.
- The loss is in normalized units: 0.1 corresponds to roughly 0.3 standard deviations of error per joint.
- **L2 or L1?** L2 penalizes large errors heavily. When demonstrations take different routes (left or right around an obstacle), L2 averages them into a path down the middle. ACT uses L1. Switching is a one-word change (`F.l1_loss`).

`get_optim_params` returns only the parameters with `requires_grad=True`, so the frozen DINOv3 weights are never passed to AdamW.

---

## 8. Asynchronous inference

`async_runner.py` deploys the policy using three independent clocks. Inference runs in a background thread, so the arm keeps moving along its current path while the next one is computed. A new plan replaces the old one as soon as it's ready, not after a fixed number of actions.

```
time ─────────────────────────────────────────────────────────────────────────►
observe  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │      obs_hz (= dataset fps, 30)
plan     ▲ submit           ▲ submit           ▲ submit          replan_hz (e.g. 5)
           └──── inference ────┐ └── inference ──┐
execute  ● ● ● ● ● ● ● ● ● ● ●│● ● ● ● ● ● ● ● ●│● ● ● ● ●      control_hz (= dataset fps, 30)
         path 0                ↑ swap to path 1   ↑ swap to path 2
```

### 8.1 The three clocks (`run()`)

All three share one loop and use `time.perf_counter()`. Each has its own `next_*` deadline. If a deadline is missed, the clock skips ahead to the next future tick instead of firing several times to catch up.

| Clock | Rate | Each tick |
|---|---|---|
| **Observation** | `--obs-hz` (default: dataset fps) | `robot.get_observation()` → `build_dataset_frame` → `prepare_observation_for_inference` → preprocessor → append to `ObservationBuffer` with a timestamp |
| **Planning** | `--replan-hz` (default 5) | Snapshot the most recent window and `planner.submit()` it. This doesn't block. |
| **Execution** | `--control-hz` (default: dataset fps) | Sample the current `Plan` at the current time → `robot.send_action()` |

Before each execution tick, the loop checks for a finished plan and switches to it if its observation is newer than the current plan's. It never switches back to an older plan.

### 8.2 `ObservationBuffer`: most recent frames

- A `deque(maxlen=n_obs_steps)` of preprocessed frames, each a `(1, …)` tensor already on the GPU, together with their capture times.
- `snapshot()` stacks the frames into `(1, T, …)` and returns the timestamp of the newest frame (`t0`).
- At the start of an episode, it fills the missing past frames with copies and builds an `observation.state_is_pad` mask, exactly as the dataset loader does. This keeps inference consistent with training.
- **Observations are captured at the training frame spacing, independently of planning.** The model was trained on frames `1/fps` apart, so the window must be built from frames `1/fps` apart, however often the policy re-plans. If you feed the model only the frames captured at each plan request, the spacing becomes `1/replan_hz` (for example 0.2 s instead of 0.033 s), which is a different input from what it was trained on. The runner prints a warning if `obs_hz` differs from the dataset fps.

### 8.3 `AsyncPlanner`: background inference

- A daemon thread that waits on a `threading.Event`.
- `submit(window, t0)` stores the request and wakes the thread. **The latest request wins:** if a request is still waiting when a newer one arrives, the newer one replaces it, so the model always plans from the freshest observation and requests never pile up.
- The worker runs `policy.predict_action_chunk(window)` under `inference_mode`, giving a normalized `(horizon, 6)` chunk. It then calls `postprocessor(chunk)`, which unnormalizes the actions, treating the horizon as the batch dimension, and produces a `Plan`.
- `pop_latest_plan()` returns the newest finished plan, or `None`. If the worker raised an exception, the exception is re-raised in the main loop instead of failing silently.
- Only the main thread talks to the robot, because the serial bus isn't thread-safe. The planner thread only uses the GPU.

### 8.4 `Plan.sample(now)`: following the path by time

Action `k` of a plan is meant for time `t0 + k·dt`, where `t0` is the capture time of the newest frame in the window and `dt = 1/fps`, the action spacing in the dataset.

```python
s = (now - t0) / dt                  # position along the path, in action steps
i, w = int(s), s - int(s)
action = (1 - w) * actions[i] + w * actions[i + 1]   # linear interpolation
```

- **Latency compensation happens automatically.** A plan that arrives 100 ms after its observation starts about 3 actions in, not at action 0, so the arm doesn't replay movements that were meant for the past.
- **`control_hz` and the training fps are independent.** Interpolation lets you execute faster or slower than the fps the actions were recorded at.
- **If the path runs out** before a new plan arrives, the arm holds the last target and the `path_ran_out` counter increases.

### 8.5 Measured behaviour

Tested with a simulated robot and real DINOv3 on the GPU, at `obs_hz=30`, `replan_hz=5` and `control_hz=30`:

| Metric | Value |
|---|---|
| Planning latency, first call (CUDA warm-up) | ~440 ms |
| Planning latency, steady state | ~100 ms |
| Execution period | 33.3 ms (target 33.3 ms) |
| `path_ran_out` | 0 |

The arm doesn't move until the first plan arrives, so the warm-up only delays the start.

### 8.6 Runner output

```
[  3.0s] obs=78 plans=10 actions=50 latency=176ms path_ran_out=0
```

`obs`, `plans` and `actions` are cumulative counts. `latency` is the mean over the last 50 plans. If `path_ran_out` is above 0, the plans are too short for the current replan rate and latency (see [§9](#9-tuning-guide)).

---

## 9. Tuning guide

### Horizon vs. replan rate

Each plan has to last until the next one arrives:

```
horizon / fps  >  1 / replan_hz  +  latency
```

At 5 Hz with about 100 ms latency, that means more than 0.3 s, so `horizon ≥ 10` at 30 fps. A `horizon` of 16 (0.53 s) leaves margin; 50 (1.67 s) is generous. Shorter horizons are easier to learn and react faster, while longer ones are more robust to slow inference. `action_queries` has one learned query per horizon step, so **changing `horizon` requires retraining**.

### Replan rate

Higher rates react faster to disturbances but use more GPU. In the steady state the planner takes about 100 ms, so rates above about 8 Hz just mean more requests get replaced while waiting. That's harmless, but it brings no gain.

### Observation window

`n_obs_steps=4` at 30 fps covers only about 0.1 s, which is enough to sense velocity. For longer context without more tokens, use a stride. For example, `observation_delta_indices = [-30, -20, -10, 0]` covers 1 s. If you do, the runner's `obs_hz` and buffer have to use the same spacing (capture every frame, but keep every 10th).

### Image tokens

| Pooling | Tokens per camera | `n_tokens_per_step` (2 cameras) | Notes |
|---|---|---|---|
| none (14×14) | 196 | 393 | Expensive; not worth it with few demonstrations |
| 2×2 max (current) | 49 | 99 | Good default |
| adaptive 4×6 | 24 | 49 | Uneven windows; coarser |
| 224×288 input, 2×2 | 63 | 127 | Keeps the 4:3 aspect ratio |

### Safety on the real arm

For the first runs, set `max_relative_target` in the `SO101FollowerConfig` in `async_runner.main()`. It caps how far a joint can move in a single command, which guards against a large sudden move from an undertrained policy.

---

## 10. Known limitations and TODOs

- **`select_action` isn't implemented.** That means synchronous `lerobot-eval` / `lerobot-record --policy` won't work yet; deploy with `async_runner.py`. To implement it, keep `_obs_queue` and `_action_queue` per step (both are created in `reset()`).
- **Fields without type annotations.** `dino_model`, `freeze_encoder`, `n_tokens_per_step`, `n_layers` and `model_dim` can't be set from the command line or saved to `config.json`. Add annotations (`n_layers: int = 4`). `model_dim` isn't used at all; the encoder reads 384 from DINOv3's config.
- **`n_tokens_per_step` is hardcoded to 99.** Adding or removing a camera, or changing the pooling, breaks it silently. Derive it instead: `len(config.image_features) * 49 + 1`.
- **`MineModel` is built twice** in `MinePolicy.__init__` (two identical lines), so DINOv3 is loaded twice at start-up. Delete one of the lines.
- **Datasets have different fps.** `so101_pick_cube_20260921_200844` is 30 fps and `…_201244` is 15 fps. The delta indices count frames, so the same window covers twice as much time in the 15 fps dataset. Don't mix them without resampling.
- **Small dataset.** Each recording has only 3 episodes. That's enough to test the pipeline, but a transformer policy needs roughly 50 or more demonstrations to learn the task.
- **Video decoding falls back to `pyav`.** `torchcodec` can't find the system FFmpeg libraries (`libavutil.so.*`). Training still works, but data loading is slower. Install FFmpeg (`sudo apt install ffmpeg`) to use the faster decoder.
- **No overlap blending at plan swaps.** The runner switches paths instantly. If the swaps look jerky, blend the first few overlapping actions of the new plan with the old one, as LeRobot's `aggregate_fn` does in its async client.
- **The deployment script hasn't been tested on hardware.** The runner was verified with a simulated robot and a real checkpoint; the SO-101 and camera setup in `main()` has not been run yet.
