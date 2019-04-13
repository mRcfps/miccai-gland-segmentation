import os

import numpy as np
from skimage.io import imsave

from keras.callbacks import Callback


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

        for idx, (img, mask) in enumerate(zip(self.x_val, self.y_val)):
            mask = self.transform_mask(mask)
            imsave(os.path.join(self.viz_dir, '{}-orig.png'.format(idx)), img)
            imsave(os.path.join(self.viz_dir, '{}-gt.png'.format(idx)), mask)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.freq == 0:
            print('Saving predicted mask to {}.'.format(self.viz_dir))
            y_pred = self.model.predict(self.x_val)
            for idx, mask in enumerate(y_pred):
                mask = self.transform_mask(mask)
                imsave(os.path.join(self.viz_dir,
                                    '{}-{}.png'.format(idx, epoch)), mask)

    def transform_mask(self, mask):
        """Transform predict mask to grayscale images."""

        n_channels = mask.shape[-1]
        if n_channels > 1:
            mask = mask.argmax(axis=-1)
        else:
            mask = mask[..., 0]

        return (mask / mask.max()).astype('float32')
