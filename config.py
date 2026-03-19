#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()

        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs1'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = True
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.TRAIN_PS = 64
        self._C.TRAINING.VAL_PS = 64

        self._C.TESTING = CN()
        self._C.TESTING.RESUME = True

        # ------------------------------
        # dataset / path
        # ------------------------------
        self._C.result_dir = './output'

        self._C.father_train_path_npz = './Datasets/GoPro/train'
        self._C.father_val_path_npz = './Datasets/GoPro/test'
        self._C.father_test_path_npz = './Datasets/GoPro/test'

        self._C.father_train_path_h5 = ''
        self._C.father_val_path_h5 = ''
        self._C.father_test_path_h5 = ''

        self._C.train_iters = 1000
        self._C.unrolling_len = 1
        self._C.img_type = ''
        self._C.event_type = ''
        self._C.pair = True
        self._C.geometry_aug = False
        self._C.pre_acc_ratio_train_dynamic = True
        self._C.compute_voxel_grid_on_cpu = True
        self._C.num_bins = 6
        self._C.num_img_bins = 1
        self._C.num_event_bins = 2

        # ------------------------------
        # model
        # ------------------------------
        self._C.skip_type = 'sum'
        self._C.activation = 'relu'
        self._C.recurrent_block_type = 'convlstm'
        self._C.num_encoders = 4
        self._C.base_num_channels = 16
        self._C.use_upsample_conv = False
        self._C.norm = 'BN'
        self._C.rec_channel = 1
        self._C.num_residual_blocks = 2
        self._C.num_output_channels = 3
        self._C.hot_pixels_file = None
        self._C.norm_method = 'normal'
        self._C.no_normalize = False
        self._C.flip = False
        self._C.VGGLayers = [1, 2, 3, 4]
        self._C.w_VGG = 0.2

        self._C.rgb_range = 255
        self._C.n_resblocks = 19
        self._C.n_feats = 64
        self._C.kernel_size = 5
        self._C.n_scales = 1

        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)
        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()