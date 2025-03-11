"""
Module for FailGenEnvWrapper.

This module provides a wrapper around the RLBench environment for generating failure cases and demos.
It supports functionalities such as recording videos, saving camera data, handling environment resets,
and managing demonstration data.

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""

import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from moviepy import ImageSequenceClip
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

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
    """
    A wrapper for the RLBench environment that integrates failure generation, recording, and demonstration saving.

    This class handles environment initialization, task setup, recording video and camera data, and
    managing failure/success demos for a given task. It uses a variety of configurations and manages internal
    caches for video frames, keyframes, and camera recordings.
    
    Attributes:
        _task_name (str): Name of the task.
        _task_folder (str): Folder containing the task files.
        _record (bool): Flag to enable video recording.
        _custom_savepath (str): Custom path to save outputs.
        _save_keyframes_only (bool): Flag to indicate if only keyframe images should be saved.
        _config (DictConfig): Loaded configuration for the task.
        _savepath (str): Final path where demos and recordings will be saved.
        _env (Environment): The RLBench environment instance.
        _task_env (Task): The specific task environment.
        _obj_base: Base object of the task.
        _manager (Manager): Manager instance for failure injection.
        _keypoints_frames (List[int]): List of step indices at which keypoints were recorded.
        _keypoints_frames_dict (Dict[str, int]): Dictionary mapping waypoint names to step indices.
        _step_counter (int): Counter tracking the number of steps taken.
        _cache_video (List[np.ndarray]): Cache of recorded video frames.
        _record_camera (Optional[VisionSensor]): Camera used for recording a cinematic video.
        _record_motion (Optional[ICameraMotion]): Camera motion controller.
        _cam_cinematic_base (Optional[Dummy]): Base dummy object for cinematic camera movement.
        _cam_base_start_pose (Optional[np.ndarray]): Initial pose of the cinematic camera base.
        _cache_cameras (dict): Dictionary caching frames from various camera views.
        _keyframe_cameras (dict): Dictionary caching keyframe images for various camera views.
    """

    def __init__(
        self,
        task_name: str,
        task_folder: str = RLBENCH_TASKPY_FOLDER,
        headless: bool = True,
        record: bool = False,
        save_data: bool = True,
        no_failures: bool = False,
        save_path: str = "",
        save_keyframes_only: bool = False,
    ):
        """
        Initialize the FailGenEnvWrapper.

        Args:
            task_name (str): Name of the task to be executed.
            task_folder (str, optional): Folder where the task modules reside. Defaults to RLBENCH_TASKPY_FOLDER.
            headless (bool, optional): Run the environment in headless mode. Defaults to True.
            record (bool, optional): Enable recording of a cinematic video. Defaults to False.
            save_data (bool, optional): Flag to determine if high-dimensional data should be saved. Defaults to True.
            no_failures (bool, optional): If True, disables failure injection. Defaults to False.
            save_path (str, optional): Custom save path for outputs. Defaults to an empty string.
            save_keyframes_only (bool, optional): If True, only keyframe images are saved rather than full videos. Defaults to False.
        """
        self._task_name: str = task_name
        self._task_folder: str = task_folder
        self._record: bool = record
        self._custom_savepath: str = save_path
        self._save_keyframes_only: bool = save_keyframes_only

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

        self._keyframe_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
            wrist=[],
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
        """
        Get the configuration for the current task.

        Returns:
            DictConfig: The OmegaConf configuration loaded for the task.
        """
        return cast(DictConfig, self._config)

    @property
    def manager(self) -> Manager:
        """
        Get the Manager responsible for failure injection.

        Returns:
            Manager: The failure injection manager.
        """
        return self._manager

    @property
    def robot(self) -> Robot:
        """
        Get the robot instance from the environment.

        Returns:
            Robot: The robot object within the current RLBench scene.
        """
        assert self._env._scene is not None
        return self._env._scene.robot

    def reset(self):
        """
        Reset the environment, manager, and cached data.

        This method resets the step counter, clears keypoints and video caches, resets the manager,
        and resets the task environment and camera poses to their initial state.
        """
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

        self._keyframe_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
            wrist=[],
        )

        if self._cam_cinematic_base and self._cam_base_start_pose is not None:
            self._cam_cinematic_base.set_pose(self._cam_base_start_pose)

    def shutdown(self) -> None:
        """
        Shutdown the RLBench environment.

        This method ensures that the environment is properly closed.
        """
        self._env.shutdown()

    def get_success(self) -> Optional[Demo]:
        """
        Retrieve a successful demonstration from the task environment.

        The method requests one successful demo, running the task with a callback at each step.

        Returns:
            Optional[Demo]: A Demo instance representing a successful run, or None if unsuccessful.
        """
        (demo,) = self._task_env.get_demos(
            amount=1,
            live_demos=True,
            max_attempts=MAX_FAILURE_ATTEMPTS,
            callable_each_step=self.on_env_step,
        )

        return demo

    def get_failure(self) -> Tuple[Optional[Demo], bool]:
        """
        Retrieve a failure demonstration from the task environment.

        The method attempts to generate a failure demo while executing various callbacks during the task.
        In case of maximum failure attempts, it prints a warning and retries.

        Returns:
            Tuple[Optional[Demo], bool]: A tuple containing the Demo instance (or None) and a success flag.
        """
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
        """
        Save a failure demonstration to disk.

        The demo is saved under a structured folder hierarchy based on the task name and episode index.

        Args:
            ep_idx (int): The episode index.
            demo (Demo): The demonstration instance to be saved.
        """
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
        """
        Save an extended failure demonstration with failure type and optional waypoint index.

        The demo is saved in a folder that distinguishes the failure type and, if applicable, the waypoint index.

        Args:
            ep_idx (int): The episode index.
            fail_type (str): A string representing the type of failure.
            demo (Demo): The demonstration instance to be saved.
            wp_idx (int, optional): Waypoint index if the failure is associated with a specific waypoint. Defaults to -1.
        """
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
        """
        Save the recorded video from cached frames to a file.

        If saving only keyframes is enabled, this method will do nothing.
        Otherwise, it renders the cached video frames into a video clip and writes it to disk.

        Args:
            filename (str): The name of the video file to save.
        """
        if self._save_keyframes_only:
            return

        if len(self._cache_video) > 0:
            check_and_make(self._savepath)
            rendered_clip = ImageSequenceClip(self._cache_video, fps=30)
            rendered_clip.write_videofile(
                os.path.join(self._savepath, filename)
            )

    def save_cameras(self, ep_idx: int, fail_type: str, wp_idx: int = -1) -> None:
        """
        Save videos from multiple camera views to disk.

        This method saves camera recordings from various perspectives into separate video files for a given episode.

        Args:
            ep_idx (int): The episode index.
            fail_type (str): A string representing the type of failure.
            wp_idx (int, optional): Waypoint index if the failure is associated with a specific waypoint. Defaults to -1.
        """
        if self._save_keyframes_only:
            return

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

    def save_keyframe_data(self, ep_idx: int, fail_type: str, wp_idx: int = -1) -> None:
        """
        Save keyframe images from camera recordings to disk.

        This method is used when only keyframe data is to be saved (i.e., not full videos). It writes
        the keyframe images for selected camera views as PNG files.

        Args:
            ep_idx (int): The episode index.
            fail_type (str): A string representing the type of failure.
            wp_idx (int, optional): Waypoint index if the failure is associated with a specific waypoint. Defaults to -1.
        """
        if not self._save_keyframes_only:
            return

        if wp_idx == -1:
            task_savepath = os.path.join(self._custom_savepath, f"{self._task_name}_{fail_type}_episode{ep_idx}")
        else:
            task_savepath = os.path.join(self._custom_savepath, f"{self._task_name}_{fail_type}_wp{wp_idx}_episode{ep_idx}")

        check_and_make(task_savepath)

        for cam_name in ["front", "overhead", "wrist"]:
            for idx, frame in enumerate(self._keyframe_cameras[cam_name]):
                pil_image = Image.fromarray(frame)
                pil_image.save(os.path.join(task_savepath, f"{cam_name}_{idx}.png"))

    def on_env_start(self, task: Task) -> None:
        """
        Callback executed at the start of the environment run.

        This method triggers the manager's start handling for the given task.
        
        Args:
            task (Task): The task instance that is starting.
        """
        self._manager.on_start(task)

    def on_env_end(self, _) -> None:
        """
        Callback executed at the end of the environment run.

        Currently a placeholder for any cleanup or final actions that need to occur when the task ends.
        """
        # if len(self._cache_video) > 0 and not success:
        #     check_and_make(self._savepath)
        #     rendered_clip = ImageSequenceClip(self._cache_video, fps=30)
        #     rendered_clip.write_videofile(
        #         os.path.join(self._savepath, f"vid_{self._task_name}.mp4")
        #     )
        ...

    def on_env_reset(self) -> None:
        """
        Callback executed on environment reset.

        This method clears caches, resets the step counter and manager, and resets camera positions.
        """
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

        self._keyframe_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
            wrist=[],
        )

        if self._cam_cinematic_base and self._cam_base_start_pose is not None:
            self._cam_cinematic_base.set_pose(self._cam_base_start_pose)

    def on_env_waypoint(self, point: Waypoint) -> None:
        """
        Callback executed at a waypoint during the task.

        This method triggers the manager's waypoint handling. Additionally, if only keyframe data is saved,
        it captures images from multiple camera views.
        
        Args:
            point (Waypoint): The current waypoint in the task.
        """
        self._manager.on_waypoint(point)

        if self._save_keyframes_only:
            obs = self._task_env.get_observation()
            self._keyframe_cameras["front"].append(obs.front_rgb)
            self._keyframe_cameras["overhead"].append(obs.overhead_rgb)
            self._keyframe_cameras["left_shoulder"].append(obs.left_shoulder_rgb)
            self._keyframe_cameras["right_shoulder"].append(obs.right_shoulder_rgb)
            self._keyframe_cameras["wrist"].append(obs.wrist_rgb)

    def on_env_waypoint_end(self, point: Waypoint) -> None:
        """
        Callback executed at the end of a waypoint.

        This method records the current step counter as a keypoint and maps the waypoint's name to this step.
        
        Args:
            point (Waypoint): The waypoint that has just ended.
        """
        self._keypoints_frames.append(self._step_counter)
        self._keypoints_frames_dict[
            point._waypoint.get_name()
        ] = self._step_counter

    def on_env_step(self, obs: Observation) -> None:
        """
        Callback executed on each environment step.

        This method increments the step counter, notifies the manager of the step, caches camera frames,
        and updates the cinematic recording if enabled.
        
        Args:
            obs (Observation): The observation data from the current environment step.
        """
        self._step_counter += 1
        self._manager.on_step()

        if self._save_keyframes_only:
            return

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
