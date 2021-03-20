"""
Replace Atari background with random videos.
Based on https://github.com/facebookresearch/deep_bisim4control/tree/master/dmc2gym
"""
import random

import cv2
import numpy as np
import skvideo.io
import tensorflow as tf
import tqdm


class RandomVideoSource:
    def __init__(
        self,
        filelist,
    ):
        """
        Args:
            filelist: a list of video files
        """
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        shape = (84, 84)
        self.total_frames = 0
        self.arr = None
        random.shuffle(self.filelist)
        for fname in tqdm.tqdm(self.filelist, desc="Loading videos", position=0):
            frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})

            local_arr = np.zeros((frames.shape[0], shape[0], shape[1]))
            for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                local_arr[i] = cv2.resize(frames[i], (shape[1], shape[0]))
            if self.arr is None:
                self.arr = local_arr
            else:
                self.arr = np.concatenate([self.arr, local_arr], 0)
            self.total_frames += local_arr.shape[0]
        self.arr = tf.convert_to_tensor(self.arr * 0.6, dtype=tf.uint8)

    def reset(self):
        self._loc = tf.Variable(
            initial_value=np.random.randint(0, self.total_frames), dtype=tf.int32
        )

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc.assign_add(1)
        return img

    def get_image_batch(self, n=256):
        imgs = []
        for _ in range(n):
            imgs.append(self.get_image())
        return tf.stack(imgs)

    def replace_background(self, background_color, frame: np.array) -> None:
        # if vectorized
        if len(frame.shape) == 4:
            stack = []
            for i in range(frame.shape[3]):
                mask = frame[..., i] == background_color
                update_values = self.get_image_batch(n=frame.shape[0])[mask]
                indices = tf.where(mask)
                if indices.shape[1] > 0:
                    stack.append(
                        tf.tensor_scatter_nd_update(
                            frame[..., i], indices, update_values
                        )
                    )
                else:
                    stack.append(frame[..., i])
            return tf.stack(stack, axis=-1)

        else:

            stack = []
            for i in range(frame.shape[2]):
                mask = frame[..., i] == background_color
                indices = tf.where(mask)
                if indices.shape[1] > 0:
                    stack.append(
                        tf.tensor_scatter_nd_update(
                            frame[..., i], indices, self.get_image()[mask]
                        )
                    )
                else:
                    stack.append(frame[..., i])
            return tf.stack(stack, axis=-1)
