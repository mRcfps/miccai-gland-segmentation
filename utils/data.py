"""
Utilities for loading training and validation data used in training.
"""

import glob
import os
import random

import numpy as np
import Augmentor
from skimage.io import imread

import config


def get_training_generator(train_dir, batch_size):
    """Training data generator with augmentation on the fly."""

    collated_images = list(zip(
        glob.glob(os.path.join(train_dir, 'images', '*.bmp')),
        glob.glob(os.path.join(train_dir, 'masks', '*.bmp')),
    ))

    data = [[imread(y) for y in x] for x in collated_images]

    p = Augmentor.DataPipeline(data)
    p.set_seed(0)
    p.rotate_random_90(0.5)
    p.zoom_random(0.5, percentage_area=0.8)
    p.crop_by_size(1, config.PATCH_SIZE, config.PATCH_SIZE, centre=False)
    p.flip_left_right(0.5)
    p.flip_top_bottom(0.5)

    def datagen():
        while True:
            augmented = p.sample(batch_size)
            x_batch = np.array([pair[0] for pair in augmented]) / 255.
            y_batch = np.expand_dims(
                np.array([pair[1] for pair in augmented]), -1)

            yield x_batch, y_batch

    return datagen()


def get_val_data(val_dir):
    x_val = np.load(os.path.join(val_dir, 'images.npy')) / 255.
    y_val = np.load(os.path.join(val_dir, 'masks.npy'))

    return x_val, y_val
