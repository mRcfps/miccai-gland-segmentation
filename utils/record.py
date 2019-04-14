"""
Utilities for recording multiple runs of experiments.
"""

import json
import os
from datetime import datetime

import config


def prepare_record_dir():
    if not os.path.exists('records'):
        os.mkdir('records')

    record_dir = os.path.join(
        'records', datetime.now().strftime('%Y%m%d-%I%M-%p'))
    os.mkdir(record_dir)
    os.mkdir(os.path.join(record_dir, 'checkpoints'))

    return record_dir


def save_params(record_dir, args):
    """Save experiment parameters to record directory."""

    args = vars(args)

    # save all parameters in config.py
    for cfg_key in dir(config):
        if not cfg_key.startswith('__'):
            args[cfg_key.lower()] = getattr(config, cfg_key)

    with open(os.path.join(record_dir, 'params.json'), 'w') as fp:
        json.dump(args, fp, indent=4)
