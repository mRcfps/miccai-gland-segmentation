"""
A collection of custom keras callbacks.
"""

import os
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave

from keras.callbacks import Callback


class LearningCurveVisualization(Callback):
    """Keras callback for plotting learning curves."""

    def __init__(self, record_dir, save_freq=1):
        self.record_dir = record_dir
        self.save_freq = save_freq
        self.history = defaultdict(list)

        self.curve_dir = os.path.join(record_dir, 'curves')
        if not os.path.exists(self.curve_dir):
            os.mkdir(self.curve_dir)

    def on_epoch_end(self, epoch, logs):
        for k, v in logs.items():
            self.history[k].append(v)

        if epoch % self.save_freq == 0:
            self._save_curves()

    def _save_curves(self):
        for key in self.history.keys():
            if key.startswith('val_'):
                continue
            plt.figure(dpi=200)

            try:
                plt.plot(self.history[key])
                plt.plot(self.history['val_' + key])
            except KeyError:
                pass

            plt.title('Model ' + key)
            plt.ylabel(key.capitalize())
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Val'])
            plt.grid(True)
            plt.savefig(os.path.join(self.curve_dir, '{}.png'.format(key)))


class MaskVisualization(Callback):
    """Keras callback for visualizing mask predictions."""

    def __init__(self, model, record_dir, x_val, y_val, freq=5):
        super().__init__()
        self.model = model
        self.x_val = x_val
        self.y_val = y_val
        self.freq = freq
        self.viz_dir = os.path.join(record_dir, 'maskviz')

        if not os.path.exists(self.viz_dir):
            os.mkdir(self.viz_dir)

        print(
            'Saving validation images and ground truth masks to {} ...'.format(self.viz_dir))
        for idx, (img, mask) in enumerate(zip(self.x_val, self.y_val)):
            img = (img * 255).astype('uint8')
            mask = self._transform_mask(mask).astype('uint8')
            imsave(os.path.join(self.viz_dir, '{}-orig.png'.format(idx)), img)
            imsave(os.path.join(self.viz_dir, '{}-gt.png'.format(idx)), mask)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.freq == 0:
            print('Saving predicted mask to {} ...'.format(self.viz_dir))
            y_pred = self.model.predict(self.x_val)
            for idx, mask in enumerate(y_pred):
                mask = self._transform_mask(mask)
                imsave(os.path.join(self.viz_dir,
                                    '{}-{}.png'.format(idx, epoch)), mask)

    def _transform_mask(self, mask):
        """Transform predict mask to grayscale images."""

        n_channels = mask.shape[-1]
        if n_channels > 1:
            mask = mask.argmax(axis=-1)
        else:
            mask = mask[..., 0]

        return (mask * 255 / mask.max()).astype('uint8')
