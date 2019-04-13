import argparse
import os
import math

# comment out next three lines if you have access to unlimited GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss

import config
from data import get_training_generator, get_val_data
from metrics import iou_score
from utils import record


def build_cli_parser():
    parser = argparse.ArgumentParser('GlaS model training script')
    parser.add_argument('dataset_path', help='Path to prepared dataset')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='# Training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--start-lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='Epoch at which to start training (useful for resuming a previous run)')
    parser.add_argument('-m', '--message', help='Note on this run')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    input_shape = (config.PATCH_SIZE, config.PATCH_SIZE, 3)
    model = Unet('densenet121', input_shape=input_shape)

    model.compile(Adam(args.start_lr), loss=bce_jaccard_loss,
                  metrics=['accuracy', iou_score])

    record_dir = record.prepare_record_dir()
    record.save_params(record_dir, args)

    checkpoint_path = os.path.join(record_dir, 'checkpoints',
                                   'weights.{epoch:03d}.hdf5')
    callbacks = [
        ModelCheckpoint(checkpoint_path, period=5,
                        save_weights_only=True, verbose=1),
        CSVLogger(os.path.join(record_dir, 'history.csv'), append=True),
        ReduceLROnPlateau('loss', factor=0.5, min_lr=1e-5, verbose=1),
        TensorBoard(os.path.join(record_dir, 'log')),
    ]

    train_datagen = get_training_generator(
        os.path.join(args.dataset_path, 'train'), args.batch_size)
    val_data = get_val_data(os.path.join(args.dataset_path, 'val'))
    steps_per_epoch = int(
        math.ceil(config.PATCHES_PER_EPOCH / args.batch_size))

    model.fit_generator(train_datagen,
                        validation_data=val_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs,
                        callbacks=callbacks)
