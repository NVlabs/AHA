# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import abc
import importlib.util
import os
import pickle
import sys
import warnings
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend import const, utils
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig


def name_to_class(task_name: str, tasks_py_folder: str) -> Optional[type]:
    """
    Returns a class for the corresponding task located at the given folder

    Parameters
    ----------
        task_name: str
            The name of the task we want to use
        tasks_py_folder: str
            The folder where to find the tasks files

    Returns
    -------
        Optional[type]
            The class found for the requested task, otherwise None
    """
    task_filepath = os.path.join(tasks_py_folder, task_name) + ".py"
    class_name = "".join([w[0].upper() + w[1:] for w in task_name.split("_")])
    spec = importlib.util.spec_from_file_location(task_name, task_filepath)
    if spec is None or spec.loader is None:
        warnings.warn(
            f"Couldn't create spec for module {task_name} @ {task_filepath}"
        )
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[task_name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        warnings.warn(f"No such taskfile {task_filepath}")
        return None

    try:
        task_class = getattr(module, class_name)
    except AttributeError:
        warnings.warn(
            f"Couldn't load class {class_name} from module @ {task_filepath}"
        )
        return None
    return task_class


class ObservationConfigExt(ObservationConfig):
    def __init__(self, data: DictConfig):
        super().__init__()

        self.set_all_low_dim(True)
        self.set_all_high_dim(False)

        # Left shoulder camera settings ----------------------------------------
        self.left_shoulder_camera.rgb = (
            data.images.rgb and data.cameras.left_shoulder
        )
        self.left_shoulder_camera.depth = (
            data.images.depth and data.cameras.left_shoulder
        )
        self.left_shoulder_camera.mask = (
            data.images.mask and data.cameras.left_shoulder
        )
        self.left_shoulder_camera.point_cloud = (
            data.images.point_cloud and data.cameras.left_shoulder
        )
        self.left_shoulder_camera.image_size = data.image_size
        self.left_shoulder_camera.depth_in_meters = data.depth_in_meters
        self.left_shoulder_camera.masks_as_one_channel = (
            data.masks_as_one_channel
        )
        self.left_shoulder_camera.render_mode = (
            RenderMode.OPENGL3
            if data.renderer == "opengl3"
            else RenderMode.OPENGL
        )

        # Right shoulder camera settings ---------------------------------------
        self.right_shoulder_camera.rgb = (
            data.images.rgb and data.cameras.right_shoulder
        )
        self.right_shoulder_camera.depth = (
            data.images.depth and data.cameras.right_shoulder
        )
        self.right_shoulder_camera.mask = (
            data.images.mask and data.cameras.right_shoulder
        )
        self.right_shoulder_camera.point_cloud = (
            data.images.point_cloud and data.cameras.right_shoulder
        )
        self.right_shoulder_camera.image_size = data.image_size
        self.right_shoulder_camera.depth_in_meters = data.depth_in_meters
        self.right_shoulder_camera.masks_as_one_channel = (
            data.masks_as_one_channel
        )
        self.right_shoulder_camera.render_mode = (
            RenderMode.OPENGL3
            if data.renderer == "opengl3"
            else RenderMode.OPENGL
        )

        # Overhead camera settings ---------------------------------------------
        self.overhead_camera.rgb = data.images.rgb and data.cameras.overhead
        self.overhead_camera.depth = data.images.depth and data.cameras.overhead
        self.overhead_camera.mask = data.images.mask and data.cameras.overhead
        self.overhead_camera.point_cloud = (
            data.images.point_cloud and data.cameras.overhead
        )
        self.overhead_camera.image_size = data.image_size
        self.overhead_camera.depth_in_meters = data.depth_in_meters
        self.overhead_camera.masks_as_one_channel = data.masks_as_one_channel
        self.overhead_camera.render_mode = (
            RenderMode.OPENGL3
            if data.renderer == "opengl3"
            else RenderMode.OPENGL
        )

        # Wrist camera settings ------------------------------------------------
        self.wrist_camera.rgb = data.images.rgb and data.cameras.wrist
        self.wrist_camera.depth = data.images.depth and data.cameras.wrist
        self.wrist_camera.mask = data.images.mask and data.cameras.wrist
        self.wrist_camera.point_cloud = (
            data.images.point_cloud and data.cameras.wrist
        )
        self.wrist_camera.image_size = data.image_size
        self.wrist_camera.depth_in_meters = data.depth_in_meters
        self.wrist_camera.masks_as_one_channel = data.masks_as_one_channel
        self.wrist_camera.render_mode = (
            RenderMode.OPENGL3
            if data.renderer == "opengl3"
            else RenderMode.OPENGL
        )

        # Front camera settings ------------------------------------------------
        self.front_camera.rgb = data.images.rgb and data.cameras.front
        self.front_camera.depth = data.images.depth and data.cameras.front
        self.front_camera.mask = data.images.mask and data.cameras.front
        self.front_camera.point_cloud = (
            data.images.point_cloud and data.cameras.front
        )
        self.front_camera.image_size = data.image_size
        self.front_camera.depth_in_meters = data.depth_in_meters
        self.front_camera.masks_as_one_channel = data.masks_as_one_channel
        self.front_camera.depth_in_meters = data.depth_in_meters
        self.front_camera.masks_as_one_channel = data.masks_as_one_channel
        self.front_camera.render_mode = (
            RenderMode.OPENGL3
            if data.renderer == "opengl3"
            else RenderMode.OPENGL
        )


def check_and_make(folder: str) -> None:
    os.makedirs(folder, exist_ok=True)


def save_demo(
    data_cfg: DictConfig,
    demo: Demo,
    example_path: str,
) -> None:
    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, const.LEFT_SHOULDER_RGB_FOLDER
    )
    left_shoulder_depth_path = os.path.join(
        example_path, const.LEFT_SHOULDER_DEPTH_FOLDER
    )
    left_shoulder_mask_path = os.path.join(
        example_path, const.LEFT_SHOULDER_MASK_FOLDER
    )
    right_shoulder_rgb_path = os.path.join(
        example_path, const.RIGHT_SHOULDER_RGB_FOLDER
    )
    right_shoulder_depth_path = os.path.join(
        example_path, const.RIGHT_SHOULDER_DEPTH_FOLDER
    )
    right_shoulder_mask_path = os.path.join(
        example_path, const.RIGHT_SHOULDER_MASK_FOLDER
    )
    overhead_rgb_path = os.path.join(example_path, const.OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, const.OVERHEAD_DEPTH_FOLDER
    )
    overhead_mask_path = os.path.join(example_path, const.OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, const.WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, const.WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, const.WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, const.FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, const.FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, const.FRONT_MASK_FOLDER)

    if data_cfg.images.rgb:
        if data_cfg.cameras.left_shoulder:
            check_and_make(left_shoulder_rgb_path)
        if data_cfg.cameras.right_shoulder:
            check_and_make(right_shoulder_rgb_path)
        if data_cfg.cameras.overhead:
            check_and_make(overhead_rgb_path)
        if data_cfg.cameras.wrist:
            check_and_make(wrist_rgb_path)
        if data_cfg.cameras.front:
            check_and_make(front_rgb_path)

    if data_cfg.images.depth:
        if data_cfg.cameras.left_shoulder:
            check_and_make(left_shoulder_depth_path)
        if data_cfg.cameras.right_shoulder:
            check_and_make(right_shoulder_depth_path)
        if data_cfg.cameras.overhead:
            check_and_make(overhead_depth_path)
        if data_cfg.cameras.wrist:
            check_and_make(wrist_depth_path)
        if data_cfg.cameras.front:
            check_and_make(front_depth_path)

    if data_cfg.images.mask:
        if data_cfg.cameras.left_shoulder:
            check_and_make(left_shoulder_mask_path)
        if data_cfg.cameras.right_shoulder:
            check_and_make(right_shoulder_mask_path)
        if data_cfg.cameras.overhead:
            check_and_make(overhead_mask_path)
        if data_cfg.cameras.wrist:
            check_and_make(wrist_mask_path)
        if data_cfg.cameras.front:
            check_and_make(front_mask_path)

    for i in range(len(demo)):
        obs = demo[i]

        if data_cfg.images.rgb:
            if data_cfg.cameras.left_shoulder:
                left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
                left_shoulder_rgb.save(
                    os.path.join(left_shoulder_rgb_path, const.IMAGE_FORMAT % i)
                )
            if data_cfg.cameras.right_shoulder:
                right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
                right_shoulder_rgb.save(
                    os.path.join(
                        right_shoulder_rgb_path, const.IMAGE_FORMAT % i
                    )
                )
            if data_cfg.cameras.overhead:
                overhead_rgb = Image.fromarray(obs.overhead_rgb)
                overhead_rgb.save(
                    os.path.join(overhead_rgb_path, const.IMAGE_FORMAT % i)
                )
            if data_cfg.cameras.wrist:
                wrist_rgb = Image.fromarray(obs.wrist_rgb)
                wrist_rgb.save(
                    os.path.join(wrist_rgb_path, const.IMAGE_FORMAT % i)
                )
            if data_cfg.cameras.front:
                front_rgb = Image.fromarray(obs.front_rgb)
                front_rgb.save(
                    os.path.join(front_rgb_path, const.IMAGE_FORMAT % i)
                )

        if data_cfg.images.depth:
            if data_cfg.cameras.left_shoulder:
                left_shoulder_depth = utils.float_array_to_rgb_image(
                    obs.left_shoulder_depth, scale_factor=const.DEPTH_SCALE
                )
                left_shoulder_depth.save(
                    os.path.join(
                        left_shoulder_depth_path, const.IMAGE_FORMAT % i
                    )
                )
            if data_cfg.cameras.right_shoulder:
                right_shoulder_depth = utils.float_array_to_rgb_image(
                    obs.right_shoulder_depth, scale_factor=const.DEPTH_SCALE
                )
                right_shoulder_depth.save(
                    os.path.join(
                        right_shoulder_depth_path, const.IMAGE_FORMAT % i
                    )
                )
            if data_cfg.cameras.overhead:
                overhead_depth = utils.float_array_to_rgb_image(
                    obs.overhead_depth, scale_factor=const.DEPTH_SCALE
                )
                overhead_depth.save(
                    os.path.join(overhead_depth_path, const.IMAGE_FORMAT % i)
                )
            if data_cfg.cameras.wrist:
                wrist_depth = utils.float_array_to_rgb_image(
                    obs.wrist_depth, scale_factor=const.DEPTH_SCALE
                )
                wrist_depth.save(
                    os.path.join(wrist_depth_path, const.IMAGE_FORMAT % i)
                )
            if data_cfg.cameras.front:
                front_depth = utils.float_array_to_rgb_image(
                    obs.front_depth, scale_factor=const.DEPTH_SCALE
                )
                front_depth.save(
                    os.path.join(front_depth_path, const.IMAGE_FORMAT % i)
                )

        if data_cfg.images.mask:
            if data_cfg.cameras.left_shoulder:
                left_shoulder_mask = Image.fromarray(
                    (obs.left_shoulder_mask * 255).astype(np.uint8)
                )
                left_shoulder_mask.save(
                    os.path.join(
                        left_shoulder_mask_path, const.IMAGE_FORMAT % i
                    )
                )

            if data_cfg.cameras.right_shoulder:
                right_shoulder_mask = Image.fromarray(
                    (obs.right_shoulder_mask * 255).astype(np.uint8)
                )
                right_shoulder_mask.save(
                    os.path.join(
                        right_shoulder_mask_path, const.IMAGE_FORMAT % i
                    )
                )

            if data_cfg.cameras.overhead:
                overhead_mask = Image.fromarray(
                    (obs.overhead_mask * 255).astype(np.uint8)
                )
                overhead_mask.save(
                    os.path.join(overhead_mask_path, const.IMAGE_FORMAT % i)
                )

            if data_cfg.cameras.wrist:
                wrist_mask = Image.fromarray(
                    (obs.wrist_mask * 255).astype(np.uint8)
                )
                wrist_mask.save(
                    os.path.join(wrist_mask_path, const.IMAGE_FORMAT % i)
                )

            if data_cfg.cameras.front:
                front_mask = Image.fromarray(
                    (obs.front_mask * 255).astype(np.uint8)
                )
                front_mask.save(
                    os.path.join(front_mask_path, const.IMAGE_FORMAT % i)
                )

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    with open(os.path.join(example_path, const.LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)


class ICameraMotion(abc.ABC):
    def __init__(self, cam: VisionSensor):
        self.cam: VisionSensor = cam
        self._prev_pose: np.ndarray = cam.get_pose()

    @abc.abstractmethod
    def step(self) -> None:
        ...

    def save_pose(self) -> None:
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self) -> None:
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(ICameraMotion):
    def __init__(self, cam: VisionSensor, origin: Dummy, speed: float):
        super().__init__(cam)
        self.origin: Dummy = origin
        self.speed: float = speed

    def step(self) -> None:
        self.origin.rotate([0, 0, self.speed])
