# model.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

if 'xlnet' not in sys.path:
    sys.path += ['xlnet']

import numpy as np
import tensorflow as tf

import xlnet.xlnet as xln



class Teacher(object):

    def __init__(self, flags, input_ids, seg_ids, input_mask):
        xlnet_config = xln.XLNetConfig(json_path=flags.model_config_path)
        run_config = xln.create_run_config(is_training=True, is_finetune=True, FLAGS=flags)
        self.model = xln.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=input_ids,
            seg_ids=seg_ids,
            input_mask=input_mask
        )

    def get_output(self):
        return self.model.get_pooled_out(summary_type='last')

    def get_embed_tb(self):
        return self.model.get_embedding_table()



class Student(object):

    def __init__(self, config):
        pass
