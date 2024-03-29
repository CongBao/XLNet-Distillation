# run_teacher.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import json
import random
import collections

if 'xlnet' not in sys.path:
    sys.path += ['xlnet']

import numpy as np
import tensorflow as tf
import sentencepiece as spm

import xlnet.xlnet as xln
import xlnet.classifier_utils as cutil
import xlnet.model_utils as mutil
import xlnet.prepro_utils as putil

import model



class Flag(object):

    def __init__(self):
        self.data_dir = 'imdb/'
        self.output_dir = 'output/'
        self.result_dir = 'result/'
        self.ckpt_path = 'ckpt/xlnet_model.ckpt'
        self.n_label = 2
        self.uncased = False
        self.do_train = True
        self.do_valid = True
        self.do_test = True
        self.spiece_model_file = 'ckpt/spiece.model'
        self.model_config_path = 'ckpt/xlnet_config.json'
        self.max_seq_length = 256
        self.n_train_epoch = 3
        self.train_batch_size = 16
        self.dev_batch_size = 16
        self.test_batch_size = 16
        self.save_summary_steps = 10
        self.save_checkpoints_steps = 20
        self.keep_checkpoint_max = None
        self.log_step_count_steps = 1

        self.use_tpu = False
        self.use_bfloat16 = False
        self.dropout = 0.1
        self.dropatt = 0.1
        self.init = 'normal'
        self.init_range = 0.1
        self.init_std = 0.02
        self.clamp_len = -1

        self.warmup_steps = 40
        self.learning_rate = 2e-5
        self.decay_method = 'poly'
        self.train_steps = 1000
        self.min_lr_ratio = 0.0
        self.weight_decay = 0.0
        self.num_core_per_host = 8
        self.adam_epsilon = 1e-8
        self.clip = 1.0
        self.lr_layer_decay_rate = 1.0



class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
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
                    guid=fn.replace('.txt', ''),
                    text_a=text,
                    label=label
                ))
        return examples



def tfrecord_example2feature(examples, label_list, max_seq_length, tokenizer, output_file):

    if tf.gfile.Exists(output_file):
        tf.logging.info('File {} exists, skip.'.format(output_file))

    tf.logging.info('Create new tfrecord {}.'.format(output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    for ex_idx, example in enumerate(examples):
        if ex_idx % 10000 == 0:
            tf.logging.info('Writing example {} of {}'.format(ex_idx, len(examples)))

        feature = cutil.convert_single_example(
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

def tfrecord_input_fn_builder(input_file, seq_length, batch_size, is_training):

    name2feat = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_length], tf.float32),
        'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.FixedLenFeature([], tf.int64)
    }

    tf.logging.info('Input tfrecord file {}.'.format(input_file))

    def map_fn(record):
        example = tf.parse_single_example(record, name2feat)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return example 
    
    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(10*batch_size)
        d = d.map(map_fn)
        d = d.batch(batch_size)
        return d

    return input_fn

def create_model(flags, input_ids, seg_ids, input_mask, labels, n_label, is_training):

    bs_per_core = tf.shape(input_ids)[0]
    in_ids = tf.transpose(input_ids, [1, 0])
    seg_ids = tf.transpose(seg_ids, [1, 0])
    input_mask = tf.transpose(input_mask, [1, 0])
    labels = tf.reshape(labels, [bs_per_core])
    
    xlnet_model = model.Teacher(
        flags=flags,
        input_ids=in_ids,
        seg_ids=seg_ids,
        input_mask=input_mask
    )

    output = xlnet_model.get_output()

    with tf.variable_scope("model/loss", reuse=tf.AUTO_REUSE):
        if is_training:
            output = tf.nn.dropout(output, rate=flags.dropout)

        logit = tf.keras.layers.Dense(n_label, kernel_initializer=xln._get_initializer(flags))(output)
        log_prob = tf.nn.log_softmax(logit, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=n_label, dtype=tf.float32)

        per_sample_loss = -tf.reduce_sum(one_hot_labels*log_prob, axis=-1)
        loss = tf.reduce_mean(per_sample_loss)

    return loss, log_prob, logit

def model_fn_builder(flags):
    
    def model_fn(features, labels, mode, params):

        input_ids = features['input_ids']
        seg_ids = features['segment_ids']
        input_mask = features['input_mask']
        label = features['label_ids']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, log_prob, logit = create_model(
            flags=flags,
            input_ids=input_ids,
            seg_ids=seg_ids,
            input_mask=input_mask,
            labels=label,
            n_label=flags.n_label,
            is_training=is_training
        )

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info('***************************')
            tf.logging.info('*** Trainable Variables ***')
            tf.logging.info('***************************')
            for var in tf.trainable_variables():
                tf.logging.info('  name = {0}, shape= {1}'.format(var.name, var.shape))
            train_op, _, _ = mutil.get_train_op(flags, loss)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            pred = tf.argmax(log_prob, axis=-1, output_type=tf.int32)
            acc = tf.metrics.accuracy(label, pred)
            auc = tf.metrics.auc(label, pred)
            f1s = tf.contrib.metrics.f1_score(label, pred)
            eval_metric_ops = {'acc': acc, 'auc': auc, 'f1s': f1s}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )
        else:
            pred = tf.argmax(log_prob, axis=-1, output_type=tf.int32)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'pred': pred, 'logit': logit}
            )

        return output_spec

    return model_fn

def main(FLAG):

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAG.output_dir)

    imdbp = ImdbProcessor()
    label_list = imdbp.get_labels()

    sp = spm.SentencePieceProcessor()
    sp.load(FLAG.spiece_model_file)

    def tokenize_fn(text):
        text = putil.preprocess_text(text, lower=FLAG.uncased)
        return putil.encode_ids(sp, text)

    n_train_step = None
    if FLAG.do_train:
        train_record_path = os.path.join(FLAG.output_dir, 'train.tfrecord')
        train_examples = imdbp.get_train_examples(FLAG.data_dir)
        random.shuffle(train_examples)
        tfrecord_example2feature(
            examples=train_examples,
            label_list=label_list,
            max_seq_length=FLAG.max_seq_length,
            tokenizer=tokenize_fn,
            output_file=train_record_path
        )
        n_train_step = int(len(train_examples)/FLAG.train_batch_size*FLAG.n_train_epoch)
        train_input_fn = tfrecord_input_fn_builder(
            input_file=train_record_path,
            seq_length=FLAG.max_seq_length,
            batch_size=FLAG.train_batch_size,
            is_training=True
        )
    if FLAG.do_valid:
        dev_record_path = os.path.join(FLAG.output_dir, 'dev.tfrecord')
        dev_examples = imdbp.get_dev_examples(FLAG.data_dir)
        tfrecord_example2feature(
            examples=dev_examples,
            label_list=label_list,
            max_seq_length=FLAG.max_seq_length,
            tokenizer=tokenize_fn,
            output_file=dev_record_path
        )
        dev_input_fn = tfrecord_input_fn_builder(
            input_file=dev_record_path,
            seq_length=FLAG.max_seq_length,
            batch_size=FLAG.dev_batch_size,
            is_training=False
        )
    if FLAG.do_test:
        test_record_path = os.path.join(FLAG.output_dir, 'test.tfrecord')
        test_examples = train_examples + dev_examples
        tfrecord_example2feature(
            examples=test_examples,
            label_list=label_list,
            max_seq_length=FLAG.max_seq_length,
            tokenizer=tokenize_fn,
            output_file=test_record_path
        )
        test_input_fn = tfrecord_input_fn_builder(
            input_file=test_record_path,
            seq_length=FLAG.max_seq_length,
            batch_size=FLAG.test_batch_size,
            is_training=False
        )
    
    model_fn = model_fn_builder(FLAG)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAG.output_dir,
        save_summary_steps=FLAG.save_summary_steps,
        save_checkpoints_steps=FLAG.save_checkpoints_steps,
        keep_checkpoint_max=FLAG.keep_checkpoint_max,
        log_step_count_steps=FLAG.log_step_count_steps
    )
    warm_config = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAG.ckpt_path,
        vars_to_warm_start='model/transformer/*'
    ) if FLAG.ckpt_path and FLAG.do_train else None
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        warm_start_from=warm_config
    )

    if FLAG.do_train and FLAG.do_valid:
        tf.logging.info('*******************************************')
        tf.logging.info('***** Running Training and Validation *****')
        tf.logging.info('*******************************************')
        tf.logging.info('  Train num examples = {}'.format(len(train_examples)))
        tf.logging.info('  Eval num examples = {}'.format(len(dev_examples)))
        tf.logging.info('  Train batch size = {}'.format(FLAG.train_batch_size))
        tf.logging.info('  Eval batch size = {}'.format(FLAG.dev_batch_size))
        tf.logging.info('  Num steps = {}'.format(n_train_step))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=n_train_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, start_delay_secs=0, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    if FLAG.do_test:
        tf.logging.info('***************************')
        tf.logging.info('***** Running Testing *****')
        tf.logging.info('***************************')
        tf.logging.info('  Num examples = {}'.format(len(test_examples)))
        tf.logging.info('  Batch size = {}'.format(FLAG.test_batch_size))
        tf.gfile.MakeDirs(FLAG.result_dir)
        embed_tb_path = os.path.join(FLAG.result_dir, 'embed_table.npy')
        if tf.gfile.Exists(embed_tb_path):
            tf.logging.info('File {} exists, skip.'.format(embed_tb_path))
        else:
            tb = estimator.get_variable_value('model/transformer/word_embedding/lookup_table')
            np.save(embed_tb_path, tb)
        train_out_path = os.path.join(FLAG.result_dir, 'train_res.tfrecord')
        valid_out_path = os.path.join(FLAG.result_dir, 'valid_res.tfrecord')
        if tf.gfile.Exists(train_out_path) and tf.gfile.Exists(valid_out_path):
            tf.logging.info('File {} and {} exists, skip.'.format(train_out_path, valid_out_path))
            return
        train_writer = tf.python_io.TFRecordWriter(train_out_path)
        valid_writer = tf.python_io.TFRecordWriter(valid_out_path)
        result = estimator.predict(input_fn=test_input_fn, checkpoint_path=FLAG.ckpt_path)
        for i, pred in enumerate(result):
            if i % 10000 == 0:
                tf.logging.info('Writting result [{} / {}]'.format(i, len(test_examples)))
            feature = cutil.convert_single_example(
                ex_index=10,
                example=test_examples[i],
                label_list=label_list,
                max_seq_length=FLAG.max_seq_length,
                tokenize_fn=tokenize_fn
            )
            def create_int_feature(values):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            def create_float_feature(values):
                return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            features = collections.OrderedDict()
            features['input_ids'] = create_int_feature(feature.input_ids)
            features['logit'] = create_float_feature(pred['logit'])
            if label_list is not None:
                features['label_ids'] = create_int_feature([feature.label_id])
            else:
                features['label_ids'] = create_float_feature([float(feature.label_id)])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            if i < len(train_examples):
                train_writer.write(tf_example.SerializeToString())
            else:
                valid_writer.write(tf_example.SerializeToString())
        train_writer.close()
        valid_writer.close()
