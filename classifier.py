# classifier.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import collections

import numpy as np
import tensorflow as tf

import xlnet

import model



class InputExample(object):

    def __init__(self, uid, text, label):
        self.uid = uid
        self.text = text
        self.label = label



class ImdbProcessor(object):

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test'))

    def get_labels(self):
        return ['neg', 'pos']

    def _create_examples(self, data_dir):
        examples = []
        for label in self.get_labels():
            cur_dir = os.path.join(data_dir, label)
            for fn in tf.gfile.ListDirectory(cur_dir):
                if not fn.endswith('txt'):
                    continue
                path = os.path.join(cur_dir, fn)
                with tf.gfile.Open(path) as f:
                    text = f.read().strip().replace('<br />', " ")
                examples.append(InputExample(
                    uid='uid',
                    text=text,
                    label=label
                ))
        return examples



def tfrecord_example2feature(examples, label_list, max_seq_length, tokenizer, output_file, n_passes=1):

    if tf.gfile.Exists(output_file):
        tf.logging.info('File {} exists, skip.'.format(output_file))

    tf.logging.info('Create new tfrecord {}.'.format(output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    if n_passes > 1:
        examples *= n_passes

    for ex_idx, example in enumerate(examples):
        if ex_idx % 10000 == 0:
            tf.logging.info('Writing example {} of {}'.format(ex_idx, len(examples)))

        feature = xlnet.classifier_utils.convert_single_example(
            ex_index=ex_idx,
            example=example,
            label_list=label_list,
            max_seq_length=max_seq_length,
            tokenize_fn=tokenizer
        )

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_float_feature(feature.input_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        if label_list is not None:
            features['label_ids'] = create_int_feature([feature.label_id])
        else:
            features['label_ids'] = create_float_feature([float(feature.label_id)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()

def tfrecord_input_fn_builder(input_file, seq_length, batch_size, is_training, is_regression=False):

    name2feat = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_length], tf.float32),
        'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.FixedLenFeature([], tf.int64)
    }

    if is_regression:
        name2feat['label_ids']: tf.FixedLenFeature([], tf.float32)

    tf.logging.info('Input tfrecord file {}.'.format(input_file))

    def map_fn(record):
        return tf.parse_single_example(record, name2feat)
    
    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.shuffle(buffer_size=10*batch_size)
            d = d.repeat()
        d = d.map(map_fn)
        d = d.batch(batch_size)
        return d

    return input_fn

def model_fn_builder():
    pass
