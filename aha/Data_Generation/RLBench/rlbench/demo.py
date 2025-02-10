import numpy as np


class Demo(object):

    def __init__(self, observations, random_seed=None, num_reset_attempts = None, keypoints_frames = [], keypoints_frames_dict = {}):
        self._observations = observations
        self.random_seed = random_seed
        self.variation_number = 0
        self.num_reset_attempts = num_reset_attempts
        self.keypoints_frames = keypoints_frames
        self.keypoints_frames_dict = keypoints_frames_dict

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)
