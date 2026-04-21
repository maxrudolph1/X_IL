import logging
import os
import pickle
from collections import defaultdict
from typing import Optional

import h5py
import numpy as np
import torch

from agents.utils.sim_path import sim_framework_path
from environments.dataset.base_dataset import TrajectoryDataset

log = logging.getLogger(__name__)


class LiberoFailureRolloutDataset(TrajectoryDataset):
    """LIBERO rollouts that include both successful and failed trials.

    The expected layout is:
        data_directory/
          task_name/
            rollouts.h5

    Each HDF5 file contains top-level trial groups with a rollout-level
    ``success`` attribute and chunked actions of shape ``(T, H, action_dim)``.
    """

    def __init__(
        self,
        data_directory: os.PathLike,
        task_embedding_suite: str = "libero_spatial",
        device: str = "cpu",
        obs_dim: int = 9,
        action_dim: int = 7,
        max_len_data: int = 63,
        window_size: int = 1,
        start_idx: int = 0,
        traj_per_task: int = 1000,
        include_successful: bool = True,
        include_failed: bool = True,
        action_mode: str = "chunk",
        action_horizon: int = 8,
        obs_frame_index: int = -1,
        cache_mode: str = "none",
        cache_images_max_gb: Optional[float] = None,
        use_returns: bool = False,
        discount: float = 0.99,
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        if not include_successful and not include_failed:
            raise ValueError("At least one of include_successful/include_failed must be True")

        if action_mode not in {"chunk", "first_action_sequence"}:
            raise ValueError(
                "action_mode must be either 'chunk' or 'first_action_sequence'"
            )
        if cache_mode not in {"none", "actions", "all"}:
            raise ValueError("cache_mode must be one of: none, actions, all")
        if discount < 0:
            raise ValueError("discount must be non-negative")

        self.data_dir = sim_framework_path(self.data_directory)
        self.task_embedding_suite = task_embedding_suite
        self.include_successful = include_successful
        self.include_failed = include_failed
        self.action_mode = action_mode
        self.action_horizon = action_horizon
        self.obs_frame_index = obs_frame_index
        self.start_idx = start_idx
        self.traj_per_task = traj_per_task
        self.cache_mode = cache_mode
        self.cache_images_max_gb = cache_images_max_gb
        self.use_returns = use_returns
        self.discount = discount

        task_emb_dir = sim_framework_path("task_embeddings")
        task_emb_path = os.path.join(task_emb_dir, f"{task_embedding_suite}.pkl")
        with open(task_emb_path, "rb") as f:
            self.tasks = pickle.load(f)

        self.records = []
        self._records_by_file = defaultdict(list)
        self._slice_indices_by_record = defaultdict(list)
        self._h5_cache = {}
        self._cached_actions = None
        self._cached_agentview_rgb = None
        self._cached_eye_in_hand_rgb = None
        self._cached_robot_states = None
        self.stats = {
            "num_trials": 0,
            "num_success": 0,
            "num_failure": 0,
            "num_missing_success": 0,
        }

        self._index_rollout_files()
        self.slices = self.get_slices()
        self.num_data = len(self.records)
        if cache_mode != "none":
            self._cache_selected_data()

        if not self.records:
            raise ValueError(f"No rollout trials found in {self.data_dir}")

        log.info(
            "Loaded %s rollout trials from %s: %s success, %s failure",
            len(self.records),
            self.data_dir,
            self.stats["num_success"],
            self.stats["num_failure"],
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = {}
        return state

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        if hasattr(value, "item"):
            value = value.item()
        return bool(value)

    def _index_rollout_files(self):
        task_dirs = [
            os.path.join(self.data_dir, name)
            for name in sorted(os.listdir(self.data_dir))
            if os.path.isdir(os.path.join(self.data_dir, name))
        ]

        for task_dir in task_dirs:
            task_name = os.path.basename(task_dir)
            rollout_path = os.path.join(task_dir, "rollouts.h5")
            if not os.path.exists(rollout_path):
                continue

            if task_name not in self.tasks:
                raise KeyError(
                    f"Task '{task_name}' is missing from "
                    f"task_embeddings/{self.task_embedding_suite}.pkl"
                )

            with h5py.File(rollout_path, "r") as h5_file:
                trial_names = sorted(
                    h5_file.keys(),
                    key=lambda name: int(name.split("_")[-1]),
                )
                selected = trial_names[self.start_idx : self.start_idx + self.traj_per_task]

                for trial_name in selected:
                    trial = h5_file[trial_name]
                    if "success" not in trial.attrs:
                        self.stats["num_missing_success"] += 1
                        continue

                    success = self._as_bool(trial.attrs["success"])
                    if success and not self.include_successful:
                        continue
                    if not success and not self.include_failed:
                        continue

                    actions = trial["actions"]
                    if actions.ndim != 3:
                        raise ValueError(
                            f"{rollout_path}/{trial_name}/actions has shape "
                            f"{actions.shape}, expected (T, H, action_dim)"
                        )
                    if actions.shape[-1] != self.action_dim:
                        raise ValueError(
                            f"{rollout_path}/{trial_name}/actions has action dim "
                            f"{actions.shape[-1]}, expected {self.action_dim}"
                        )

                    horizon = int(actions.shape[1])
                    if self.action_mode == "chunk" and horizon != self.action_horizon:
                        raise ValueError(
                            f"{rollout_path}/{trial_name}/actions has horizon "
                            f"{horizon}, expected {self.action_horizon}"
                        )

                    record = {
                        "path": rollout_path,
                        "trial_name": trial_name,
                        "task_name": task_name,
                        "success": success,
                        "length": int(actions.shape[0]),
                    }
                    self._records_by_file[rollout_path].append(len(self.records))
                    self.records.append(record)
                    self.stats["num_trials"] += 1
                    if success:
                        self.stats["num_success"] += 1
                    else:
                        self.stats["num_failure"] += 1

    def _get_h5_file(self, path):
        h5_file = self._h5_cache.get(path)
        if h5_file is None:
            h5_file = h5py.File(path, "r")
            self._h5_cache[path] = h5_file
        return h5_file

    def __del__(self):
        for h5_file in getattr(self, "_h5_cache", {}).values():
            try:
                h5_file.close()
            except Exception:
                pass

    def get_slices(self):
        slices = []
        slice_record_indices = []
        slice_starts = []
        for record_idx, record in enumerate(self.records):
            length = record["length"]
            if self.action_mode == "chunk":
                starts = range(length)
            else:
                if length - self.window_size < 0:
                    log.warning(
                        "Ignored short sequence #%s: len=%s, window=%s",
                        record_idx,
                        length,
                        self.window_size,
                    )
                    continue
                starts = range(length - self.window_size + 1)

            for start in starts:
                slice_idx = len(slices)
                slices.append((record_idx, start))
                slice_record_indices.append(record_idx)
                slice_starts.append(start)
                self._slice_indices_by_record[record_idx].append(slice_idx)

        self.slice_record_indices = np.asarray(slice_record_indices, dtype=np.int32)
        self.slice_starts = np.asarray(slice_starts, dtype=np.int16)
        self._slice_indices_by_record = {
            record_idx: np.asarray(indices, dtype=np.int32)
            for record_idx, indices in self._slice_indices_by_record.items()
        }
        return slices

    def _estimate_image_cache_gb(self, image_shape):
        bytes_per_camera = len(self.slices) * int(np.prod(image_shape)) * np.dtype(np.uint8).itemsize
        return (2 * bytes_per_camera) / (1024 ** 3)

    def _cache_selected_data(self):
        if not self.slices:
            return

        action_shape = self._cached_action_shape()
        self._cached_actions = np.empty(
            (len(self.slices),) + action_shape,
            dtype=np.float32,
        )

        should_cache_images = self.cache_mode == "all"
        sample_record = self.records[int(self.slice_record_indices[0])]
        sample_start = int(self.slice_starts[0])
        with h5py.File(sample_record["path"], "r") as h5_file:
            sample_trial = h5_file[sample_record["trial_name"]]
            h5_image_shape = sample_trial["obs"]["agentview_rgb"][sample_start, self.obs_frame_index].shape

        if should_cache_images:
            estimated_gb = self._estimate_image_cache_gb(h5_image_shape)
            if self.cache_images_max_gb is not None and estimated_gb > self.cache_images_max_gb:
                log.warning(
                    "Skipping image cache: estimated %.2f GiB exceeds limit %.2f GiB",
                    estimated_gb,
                    self.cache_images_max_gb,
                )
                should_cache_images = False
            else:
                log.info("Caching rollout images in RAM: estimated %.2f GiB", estimated_gb)
                cached_image_shape = (
                    h5_image_shape[2],
                    h5_image_shape[0],
                    h5_image_shape[1],
                )
                self._cached_agentview_rgb = np.empty(
                    (len(self.slices),) + cached_image_shape,
                    dtype=np.uint8,
                )
                self._cached_eye_in_hand_rgb = np.empty_like(self._cached_agentview_rgb)
                self._cached_robot_states = np.empty(
                    (len(self.slices), self.obs_dim),
                    dtype=np.float32,
                )

        log.info("Caching rollout actions in RAM")
        cached_records = 0
        for path, record_indices in self._records_by_file.items():
            with h5py.File(path, "r") as h5_file:
                for record_idx in record_indices:
                    record = self.records[record_idx]
                    trial = h5_file[record["trial_name"]]
                    slice_indices = self._slice_indices_by_record[record_idx]
                    starts = self.slice_starts[slice_indices]
                    contiguous_dst = (
                        len(slice_indices) > 0
                        and np.all(np.diff(slice_indices) == 1)
                        and np.array_equal(starts, np.arange(record["length"]))
                    )
                    if contiguous_dst:
                        dst = slice(int(slice_indices[0]), int(slice_indices[-1]) + 1)
                    else:
                        dst = slice_indices

                    trial_actions = trial["actions"][:].astype(np.float32, copy=False)
                    if self.action_mode == "chunk":
                        self._cached_actions[dst] = trial_actions if contiguous_dst else trial_actions[starts]
                    else:
                        for dst_idx, start in zip(slice_indices, starts):
                            end = int(start) + self.window_size
                            self._cached_actions[dst_idx] = trial_actions[start:end, 0, :]

                    if should_cache_images:
                        obs = trial["obs"]
                        agentview_rgb = obs["agentview_rgb"][:, self.obs_frame_index]
                        eye_in_hand_rgb = obs["eye_in_hand_rgb"][:, self.obs_frame_index]
                        joint_states = obs["joint_states"][:, self.obs_frame_index]
                        gripper_states = obs["gripper_states"][:, self.obs_frame_index]

                        if not contiguous_dst:
                            agentview_rgb = agentview_rgb[starts]
                            eye_in_hand_rgb = eye_in_hand_rgb[starts]
                            joint_states = joint_states[starts]
                            gripper_states = gripper_states[starts]

                        self._cached_agentview_rgb[dst] = np.moveaxis(agentview_rgb, -1, 1)
                        self._cached_eye_in_hand_rgb[dst] = np.moveaxis(eye_in_hand_rgb, -1, 1)
                        self._cached_robot_states[dst] = np.concatenate(
                            [joint_states, gripper_states],
                            axis=-1,
                        ).astype(np.float32, copy=False)
                    cached_records += 1
                    if cached_records % 500 == 0:
                        log.info("Cached %s rollout trials", cached_records)

    def _cached_action_shape(self):
        if self.action_mode == "chunk":
            return (self.action_horizon, self.action_dim)
        return (self.window_size, self.action_dim)

    def _return_at(self, record, timestep):
        if not record["success"]:
            return 0.0

        steps_to_terminal = max(record["length"] - 1 - int(timestep), 0)
        return float(self.discount ** steps_to_terminal)

    def _get_returns(self, record, start, action_length):
        if self.action_mode == "chunk":
            return np.full(
                (action_length,),
                self._return_at(record, start),
                dtype=np.float32,
            )

        timesteps = np.arange(start, start + action_length, dtype=np.int32)
        if not record["success"]:
            return np.zeros((action_length,), dtype=np.float32)

        steps_to_terminal = np.maximum(record["length"] - 1 - timesteps, 0)
        return np.power(self.discount, steps_to_terminal).astype(np.float32)

    def get_seq_length(self, idx):
        return int(self.records[idx]["length"])

    def get_all_actions(self):
        if self._cached_actions is not None:
            return torch.from_numpy(
                self._cached_actions.reshape(-1, self.action_dim)
            ).float()

        actions = []
        for path, record_indices in self._records_by_file.items():
            with h5py.File(path, "r") as h5_file:
                for record_idx in record_indices:
                    record = self.records[record_idx]
                    trial_actions = h5_file[record["trial_name"]]["actions"][:].astype(
                        np.float32
                    )
                    if self.action_mode == "chunk":
                        actions.append(trial_actions.reshape(-1, self.action_dim))
                    else:
                        actions.append(trial_actions[:, 0, :])

        return torch.from_numpy(np.concatenate(actions, axis=0)).float()

    def get_all_observations(self):
        raise NotImplementedError(
            "Loading every image observation from failure rollouts is intentionally "
            "disabled outside the explicit cache path."
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        record_idx = int(self.slice_record_indices[idx])
        start = int(self.slice_starts[idx])
        record = self.records[record_idx]

        if self._cached_actions is not None:
            action = self._cached_actions[idx]
            mask = np.ones((action.shape[0],), dtype=np.float32)
        else:
            trial = self._get_h5_file(record["path"])[record["trial_name"]]
            if self.action_mode == "chunk":
                action = trial["actions"][start].astype(np.float32)
                mask = np.ones((self.action_horizon,), dtype=np.float32)
            else:
                end = start + self.window_size
                action = trial["actions"][start:end, 0, :].astype(np.float32)
                mask = np.ones((self.window_size,), dtype=np.float32)

        if self._cached_agentview_rgb is not None:
            agentview_rgb = self._cached_agentview_rgb[idx]
            eye_in_hand_rgb = self._cached_eye_in_hand_rgb[idx]
            robot_states = self._cached_robot_states[idx]
        else:
            trial = self._get_h5_file(record["path"])[record["trial_name"]]
            agentview_rgb = trial["obs"]["agentview_rgb"][start, self.obs_frame_index]
            eye_in_hand_rgb = trial["obs"]["eye_in_hand_rgb"][start, self.obs_frame_index]
            joint_states = trial["obs"]["joint_states"][start, self.obs_frame_index]
            gripper_states = trial["obs"]["gripper_states"][start, self.obs_frame_index]
            robot_states = np.concatenate([joint_states, gripper_states], axis=-1).astype(
                np.float32
            )

        obs = {
            "agentview_image": self._image_to_tensor(agentview_rgb),
            "eye_in_hand_image": self._image_to_tensor(eye_in_hand_rgb),
            "lang_emb": self.tasks[record["task_name"]].float(),
            "robot_states": torch.from_numpy(robot_states).float().unsqueeze(0),
        }

        action_tensor = torch.from_numpy(action).float()
        mask_tensor = torch.from_numpy(mask).float()

        if self.use_returns:
            returns = self._get_returns(record, start, action.shape[0])
            return obs, action_tensor, mask_tensor, torch.from_numpy(returns).float()

        return obs, action_tensor, mask_tensor

    @staticmethod
    def _image_to_tensor(image):
        image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image).float()
        if image_tensor.shape[0] != 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor.unsqueeze(0) / 255.0
