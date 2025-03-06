# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from moviepy import ImageSequenceClip
from omegaconf import DictConfig, ListConfig, OmegaConf
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import const
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint
from rlbench.demo import Demo
from rlbench.environment import DIR_PATH, Environment

from failgen.fail_manager import Manager
from failgen.utils import (
    CircleCameraMotion,
    ICameraMotion,
    ObservationConfigExt,
    check_and_make,
    name_to_class,
    save_demo,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(CURRENT_DIR, "configs")
RLBENCH_TASKPY_FOLDER = os.path.join(DIR_PATH, "tasks")

MAX_FAILURE_ATTEMPTS = 5


class FailGenEnvWrapper:
    def __init__(
        self,
        task_name: str,
        task_folder: str = RLBENCH_TASKPY_FOLDER,
        headless: bool = True,
        record: bool = False,
        save_data: bool = True,
        no_failures: bool = False,
        save_path: str = "",
    ):
        self._task_name: str = task_name
        self._task_folder: str = task_folder
        self._record: bool = record
        self._custom_savepath: str = save_path

        self._config_filepath = os.path.join(CONFIGS_DIR, task_name)
        self._config = OmegaConf.load(f"{self._config_filepath}.yaml")
        if self._custom_savepath == "":
            self._savepath: str = os.path.join(
                self._config.data.save_path, self._task_name
            )
        else:
            self._savepath: str = os.path.join(
                self._custom_savepath, self._task_name
            )

        check_and_make(self._savepath)

        if no_failures:
            self._config.failures = ListConfig([])

        obs_config = ObservationConfigExt(self._config.data)
        if not save_data:
            obs_config.set_all_high_dim(False)

        self._env: Environment = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=headless,
        )
        self._env.launch()

        task_class = name_to_class(task_name, task_folder)
        if task_class is None:
            raise RuntimeError(f"Couldn't instantiate task '{task_name}'")
        self._task_env = self._env.get_task(task_class)
        self._obj_base = self._task_env._task.get_base()

        assert self._env._scene is not None
        self._manager = Manager(
            self._env._scene.robot, self._obj_base, self._config.failures
        )

        self._keypoints_frames: List[int] = []
        self._keypoints_frames_dict: Dict[str, int] = {}
        self._step_counter: int = 0

        self._cache_video: List[np.ndarray] = []

        # Create some extra resources for recording a separate video
        self._record_camera: Optional[VisionSensor] = None
        self._record_motion: Optional[ICameraMotion] = None
        self._cam_cinematic_base: Optional[Dummy] = None
        self._cam_base_start_pose: Optional[np.ndarray] = None

        self._cache_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
        )

        if self._record:
            self._cam_cinematic_base = Dummy("cam_cinematic_base")
            self._cam_base_start_pose = self._cam_cinematic_base.get_pose()
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self._record_camera = VisionSensor.create([1280, 720])
            # self._record_camera.set_explicit_handling(True)
            self._record_camera.set_pose(cam_placeholder.get_pose())
            self._record_camera.set_render_mode(RenderMode.OPENGL3)
            self._record_camera.set_parent(cam_placeholder)
            self._record_camera.set_position([1.082, -0.6550, 1.800])
            self._record_camera.set_orientation(
                (np.pi / 180.0) * np.array([-147.27, -32.798, 139.88])
            )

            self._record_motion = CircleCameraMotion(
                self._record_camera, self._cam_cinematic_base, 0.01
            )

            tf = self._record_camera.get_matrix()
            cam_pos = tf[:3, 3]
            _, _, cam_z = tf[:3, 0], tf[:3, 1], tf[:3, 2]
            new_cam_pos = cam_pos - cam_z * 1.05
            self._record_camera.set_position(new_cam_pos)

    @property
    def config(self) -> DictConfig:
        return cast(DictConfig, self._config)

    @property
    def manager(self) -> Manager:
        return self._manager

    @property
    def robot(self) -> Robot:
        assert self._env._scene is not None
        return self._env._scene.robot

    def reset(self):
        self._step_counter = 0
        self._keypoints_frames.clear()
        self._keypoints_frames_dict.clear()
        self._cache_video.clear()
        self._manager.on_reset()
        self._task_env.reset()

        self._cache_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
        )

        if self._cam_cinematic_base and self._cam_base_start_pose is not None:
            self._cam_cinematic_base.set_pose(self._cam_base_start_pose)

    def shutdown(self) -> None:
        self._env.shutdown()

    def get_success(self) -> Optional[Demo]:
        (demo,) = self._task_env.get_demos(
            amount=1,
            live_demos=True,
            max_attempts=MAX_FAILURE_ATTEMPTS,
            callable_each_step=self.on_env_step,
        )

        return demo

    def get_failure(self) -> Tuple[Optional[Demo], bool]:
        demo: Optional[Demo] = None
        try:
            (demo,), success = self._task_env.get_failures(
                amount=1,
                max_attempts=MAX_FAILURE_ATTEMPTS,
                callable_each_waypoint=self.on_env_waypoint,
                callable_each_end_waypoint=self.on_env_waypoint_end,
                callable_each_step=self.on_env_step,
                callable_each_reset=self.on_env_reset,
                callable_on_start=self.on_env_start,
                callable_on_end=self.on_env_end,
            )
        except RuntimeError:
            print(
                "Warn >>> get_failure reached max attempts. "
                + "Won't crash but will retry some more times. "
                + f"task_name: {self._task_name}"
            )
            success = True

        if demo is not None:
            demo.keypoints_frames = self._keypoints_frames.copy()
            demo.keypoints_frames_dict = self._keypoints_frames_dict.copy()
        return demo, success

    def save_failure(self, ep_idx: int, demo: Demo) -> None:
        task_savepath = os.path.join(
            self._config.data.save_path, self._task_name
        )
        check_and_make(task_savepath)

        episodes_path = os.path.join(task_savepath, const.EPISODES_FOLDER)
        check_and_make(episodes_path)

        episode_path = os.path.join(
            episodes_path, const.EPISODE_FOLDER % ep_idx
        )

        save_demo(self._config.data, demo, episode_path)

    def save_failure_ext(
        self, ep_idx: int, fail_type: str, demo: Demo, wp_idx: int = -1
    ) -> None:
        if wp_idx == -1:
            task_savepath = os.path.join(
                self._config.data.save_path, self._task_name, fail_type
            )
        else:
            task_savepath = os.path.join(
                self._config.data.save_path,
                self._task_name,
                f"{fail_type}_wp{wp_idx}",
            )
        check_and_make(task_savepath)

        episodes_path = os.path.join(task_savepath, const.EPISODES_FOLDER)
        check_and_make(episodes_path)

        episode_path = os.path.join(
            episodes_path, const.EPISODE_FOLDER % ep_idx
        )

        save_demo(self._config.data, demo, episode_path)

    def save_video(self, filename: str) -> None:
        if len(self._cache_video) > 0:
            check_and_make(self._savepath)
            rendered_clip = ImageSequenceClip(self._cache_video, fps=30)
            rendered_clip.write_videofile(
                os.path.join(self._savepath, filename)
            )

    def save_cameras(self, ep_idx: int, fail_type: str, wp_idx: int = -1) -> None:
        if wp_idx == -1:
            task_savepath = os.path.join(
                self._savepath, fail_type
            )
        else:
            task_savepath = os.path.join(
                self._savepath,
                f"{fail_type}_wp{wp_idx}",
            )
        check_and_make(task_savepath)

        episodes_path = os.path.join(task_savepath, const.EPISODES_FOLDER)
        check_and_make(episodes_path)

        episode_path = os.path.join(
            episodes_path, const.EPISODE_FOLDER % ep_idx
        )

        check_and_make(episode_path)
        for cam_name in self._cache_cameras.keys():
            clip = ImageSequenceClip(self._cache_cameras[cam_name], fps=30)
            clip.write_videofile(
                os.path.join(episode_path, f"vid_{cam_name}.mp4"), logger=None,
            )

    def on_env_start(self, task: Task) -> None:
        self._manager.on_start(task)

    def on_env_end(self, _) -> None:
        # if len(self._cache_video) > 0 and not success:
        #     check_and_make(self._savepath)
        #     rendered_clip = ImageSequenceClip(self._cache_video, fps=30)
        #     rendered_clip.write_videofile(
        #         os.path.join(self._savepath, f"vid_{self._task_name}.mp4")
        #     )
        ...

    def on_env_reset(self) -> None:
        self._step_counter = 0
        self._keypoints_frames.clear()
        self._keypoints_frames_dict.clear()
        self._cache_video.clear()
        self._manager.on_reset()

        self._cache_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
        )

        if self._cam_cinematic_base and self._cam_base_start_pose is not None:
            self._cam_cinematic_base.set_pose(self._cam_base_start_pose)

    def on_env_waypoint(self, point: Waypoint) -> None:
        self._manager.on_waypoint(point)

    def on_env_waypoint_end(self, point: Waypoint) -> None:
        self._keypoints_frames.append(self._step_counter)
        self._keypoints_frames_dict[
            point._waypoint.get_name()
        ] = self._step_counter

    def on_env_step(self, obs: Observation) -> None:
        self._step_counter += 1
        self._manager.on_step()

        self._cache_cameras["front"].append(obs.front_rgb)
        self._cache_cameras["overhead"].append(obs.overhead_rgb)
        self._cache_cameras["left_shoulder"].append(obs.left_shoulder_rgb)
        self._cache_cameras["right_shoulder"].append(obs.right_shoulder_rgb)

        if self._record_motion is not None:
            self._record_motion.step()

        if self._record_camera is not None:
            self._cache_video.append(
                np.clip(
                    (self._record_camera.capture_rgb() * 255.0).astype(
                        np.uint8
                    ),
                    0,
                    255,
                )
            )
