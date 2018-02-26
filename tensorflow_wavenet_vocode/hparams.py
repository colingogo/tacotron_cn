# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    name="wavenet_vocoder",

    NPY_DATAROOT="/mnt/lustre/sjtu/users/kc430/data/my/wavenet/cmu_arctic",

    # Audio:
    sample_rate=16000,
    silence_threshold=2,
    num_mels=80,
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,

    # global condition if False set global channel to None
    gc_enable=True,
    global_channel=16,
    global_cardinality=7,  # speaker num

    filter_width=2,
    # dilations=[1, 2, 4, 8, 16, 32, 64, 128,
    #            1, 2, 4, 8, 16, 32, 64, 128],
    dilations=[1, 2, 4, 8, 16, 32,
               1, 2, 4, 8, 16, 32,
               1, 2, 4, 8, 16, 32,
               1, 2, 4, 8, 16, 32],
    residual_channels=512,
    dilation_channels=512,
    quantization_channels=256,
    skip_channels=256,
    use_biases=True,
    scalar_input=False,
    initial_filter_width=32,

    MOVING_AVERAGE_DECAY=0.9999,

    upsample_conditional_features=True,
    upsample_factor=[16, 16],

    LEARNING_RATE_DECAY_FACTOR=0.5,
    NUM_STEPS_RATIO_PER_DECAY=0.3,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
