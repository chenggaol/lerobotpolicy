# async_runner.py
"""Asynchronous inference for MinePolicy with three independent fixed-rate clocks.

  observation clock (obs_hz):     read the robot, append the frame to a rolling window
  planning clock    (replan_hz):  hand the latest window to a background planner thread
  execution clock   (control_hz): follow the current planned path, sampled by time

The robot keeps executing the current path while the next one is being computed, and
switches to the new path as soon as it is ready.

Run on the SO-101 (from the lerobot repo root):
    python -m my_policy.async_runner --checkpoint outputs/train/.../pretrained_model \
        --dataset-repo-id chenggao/so101_pick_cube_all_v2 --dataset-root data/so101_pick_cube_all_v2 \
        --port /dev/ttyACM0 --robot-id my_awesome_follower_arm \
        --front-cam /dev/video0 --wrist-cam /dev/video2 --replan-hz 5
"""
import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame


# ---------- 1. OBSERVATION WINDOW: most recent n_obs_steps frames ----------
class ObservationBuffer:
    """Rolling window of preprocessed frames, captured at the training frame spacing."""

    def __init__(self, n_obs_steps: int, keys: list[str]):
        self.T = n_obs_steps
        self.keys = keys
        self.frames = deque(maxlen=n_obs_steps)  #maxlen drops the oldest frame automatically
        self.times = deque(maxlen=n_obs_steps)

    def reset(self):
        self.frames.clear()
        self.times.clear()

    def __len__(self):
        return len(self.frames)

    def append(self, frame: dict[str, torch.Tensor], t: float):
        #frame: each tensor (1, ...) already normalized and on the policy device
        self.frames.append({k: frame[k] for k in self.keys})
        self.times.append(t)

    def snapshot(self) -> tuple[dict[str, torch.Tensor], float]:
        """Window (1, T, ...) + obs pad mask, and the timestamp of the newest frame."""
        n_real = len(self.frames)
        #start of an episode: fill missing past frames with copies of the oldest real frame,
        #and mark them as padding, same as the dataset loader does during training
        frames = [self.frames[0]] * (self.T - n_real) + list(self.frames)
        window = {k: torch.stack([f[k] for f in frames], dim=1) for k in self.keys}
        device = window[OBS_STATE].device
        window[f"{OBS_STATE}_is_pad"] = (torch.arange(self.T, device=device) < self.T - n_real)[None]
        return window, self.times[-1]


# ---------- 2. PLANNED PATH: action chunk indexed by time ----------
@dataclass
class Plan:
    t0: float              #timestamp of the newest observation the plan was computed from
    actions: np.ndarray    #(horizon, action_dim), unnormalized joint targets
    dt: float              #time between consecutive actions (1 / dataset fps)
    latency: float         #seconds from submit to ready

    def sample(self, now: float) -> tuple[np.ndarray, bool]:
        """Linearly interpolated action at wall time `now`; second value is True past the end."""
        s = (now - self.t0) / self.dt  #action k is meant for time t0 + k*dt
        if s <= 0:
            return self.actions[0], False
        i = int(s)
        if i >= len(self.actions) - 1:
            return self.actions[-1], True  #path ran out before a new plan arrived: hold last target
        w = s - i
        return (1 - w) * self.actions[i] + w * self.actions[i + 1], False


# ---------- 3. PLANNER: runs the model in a background thread ----------
class AsyncPlanner:
    """Latest-request-wins planner: a new submit replaces any request still waiting."""

    def __init__(self, policy, postprocessor, action_dt: float):
        self.policy = policy
        self.postprocessor = postprocessor
        self.action_dt = action_dt
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._request = None
        self._plan = None
        self._error = None
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, window: dict[str, torch.Tensor], t0: float):
        with self._lock:
            self._request = (window, t0, time.perf_counter())
        self._wake.set()

    def pop_latest_plan(self) -> Plan | None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("planner thread failed") from self._error
            plan, self._plan = self._plan, None
        return plan

    def busy(self) -> bool:
        with self._lock:
            return self._request is not None

    def close(self):
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2)

    def _worker(self):
        while not self._stop:
            self._wake.wait()
            with self._lock:
                request, self._request = self._request, None
                self._wake.clear()
            if request is None:
                continue
            window, t0, submitted = request
            try:
                with torch.inference_mode():
                    chunk = self.policy.predict_action_chunk(window)[0]   # (horizon, action_dim), normalized
                    #postprocessor unnormalizes per action, so pass the horizon as the batch dim
                    actions = self.postprocessor(chunk).cpu().numpy()     # (horizon, action_dim)
            except Exception as e:
                with self._lock:
                    self._error = e
                return
            plan = Plan(t0=t0, actions=actions, dt=self.action_dt, latency=time.perf_counter() - submitted)
            with self._lock:
                self._plan = plan


# ---------- 4. CONTROL LOOP: three fixed-rate clocks ----------
def run(robot, policy, preprocessor, postprocessor, ds_features: dict, fps: float, *,
        obs_hz: float, replan_hz: float, control_hz: float, duration_s: float, log_every_s: float = 2.0):
    device = next(policy.parameters()).device
    keys = [OBS_STATE, *policy.config.image_features]
    action_names = ds_features[ACTION]["names"]
    if abs(obs_hz - fps) > 1e-6:
        print(f"WARNING: obs_hz={obs_hz} but the policy was trained on frames {1/fps:.3f}s apart ({fps} fps)")

    buffer = ObservationBuffer(policy.config.n_obs_steps, keys)
    planner = AsyncPlanner(policy, postprocessor, action_dt=1.0 / fps)
    plan: Plan | None = None
    stats = {"obs": 0, "plans": 0, "actions": 0, "exhausted": 0, "latency": deque(maxlen=50)}

    def tick(next_t: float, period: float, now: float) -> float:
        #advance to the next tick in the future, skipping missed ticks instead of bursting
        while next_t <= now:
            next_t += period
        return next_t

    start = time.perf_counter()
    next_obs = next_replan = next_control = next_log = start
    try:
        #duration_s <= 0 runs until Ctrl+C
        while (now := time.perf_counter()) - start < duration_s or duration_s <= 0:
            #observation clock: store the most recent observation in the window
            if now >= next_obs:
                raw = robot.get_observation()
                frame = build_dataset_frame(ds_features, raw, prefix=OBS_STR)
                frame = preprocessor(prepare_observation_for_inference(frame, device))
                buffer.append(frame, now)
                stats["obs"] += 1
                next_obs = tick(next_obs, 1.0 / obs_hz, now)

            #planning clock: feed the latest window to the predictor
            if now >= next_replan and len(buffer) > 0:
                window, t0 = buffer.snapshot()
                planner.submit(window, t0)
                next_replan = tick(next_replan, 1.0 / replan_hz, now)

            #swap in a new path as soon as one is ready (never go back to an older one)
            new_plan = planner.pop_latest_plan()
            if new_plan is not None and (plan is None or new_plan.t0 > plan.t0):
                plan = new_plan
                stats["plans"] += 1
                stats["latency"].append(new_plan.latency)

            #execution clock: keep following the current path until a new one arrives
            if now >= next_control and plan is not None:
                action, exhausted = plan.sample(now)
                robot.send_action({name: float(v) for name, v in zip(action_names, action)})
                stats["actions"] += 1
                stats["exhausted"] += exhausted
                next_control = tick(next_control, 1.0 / control_hz, now)

            if now >= next_log:
                lat = np.mean(stats["latency"]) * 1e3 if stats["latency"] else float("nan")
                print(f"[{now - start:5.1f}s] obs={stats['obs']} plans={stats['plans']} "
                      f"actions={stats['actions']} latency={lat:.0f}ms path_ran_out={stats['exhausted']}")
                next_log = tick(next_log, log_every_s, now)

            #sleep until the next clock is due
            time.sleep(max(0.0, min(next_obs, next_replan, next_control) - time.perf_counter()))
    finally:
        planner.close()
    return stats


# ---------- 5. ENTRY POINT: SO-101 ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="pretrained_model dir from lerobot-train")
    parser.add_argument("--dataset-repo-id", required=True, help="dataset the policy was trained on")
    parser.add_argument("--dataset-root", default=None, help="local dataset folder, e.g. data/so101_pick_cube_all_v2")
    parser.add_argument("--port", required=True, help="follower arm serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--robot-id", default=None, help="calibration id used when recording")
    parser.add_argument("--front-cam", required=True, help="camera index or path, e.g. /dev/video0")
    parser.add_argument("--wrist-cam", required=True, help="camera index or path, e.g. /dev/video2")
    parser.add_argument("--camera-fps", type=int, default=30, help="rate the cameras stream at (same as when recording)")
    parser.add_argument("--obs-hz", type=float, default=None, help="default: dataset fps")
    parser.add_argument("--replan-hz", type=float, default=5.0)
    parser.add_argument("--control-hz", type=float, default=None, help="default: dataset fps")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to run, 0 = until Ctrl+C")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    from .modeling_mine import MinePolicy

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    h, w, _ = meta.features["observation.images.front"]["shape"]

    def camera(index_or_path: str) -> OpenCVCameraConfig:
        #same settings as lerobot-record: MJPG over V4L2 so 640x480 streams at full rate
        src = int(index_or_path) if index_or_path.isdigit() else index_or_path
        return OpenCVCameraConfig(index_or_path=src, fps=args.camera_fps, width=w, height=h,
                                  fourcc="MJPG", backend=Cv2Backends.V4L2)

    cameras = {"front": camera(args.front_cam), "wrist": camera(args.wrist_cam)}
    robot = make_robot_from_config(SO101FollowerConfig(port=args.port, id=args.robot_id, cameras=cameras))

    policy = MinePolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    policy.config.device = args.device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    robot.connect()
    try:
        run(robot, policy, preprocessor, postprocessor, meta.features, meta.fps,
            obs_hz=args.obs_hz or meta.fps, replan_hz=args.replan_hz,
            control_hz=args.control_hz or meta.fps, duration_s=args.duration)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
