# run_student.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf

import model



class Flag(object):

    def __init__(self):
        self.train_data_path = 'result/train_res.tfrecord'
        self.valid_data_path = 'result/valid_res.tfrecord'
        self.test_data_path = 'result/valid_res.tfrecord'
        self.embed_tb_path = 'result/embed_table.npy'
        self.output_dir = 'output/'
        self.ckpt_path = None
        self.n_label = 2
        self.do_train = True
        self.do_valid = True
        self.do_test = True

        self.units = 768
        self.dropout = 0.1
        self.learning_rate = 1e-3
        self.n_label = 2

        self.max_seq_length = 256
        self.n_train_step = 1000
        self.train_batch_size = 32
        self.dev_batch_size = 32
        self.test_batch_size = 32
        self.save_summary_steps = 10
        self.save_checkpoints_steps = 20
        self.keep_checkpoint_max = None
        self.log_step_count_steps = 1



def tfrecord_input_fn_builder(input_file, seq_length, n_label, batch_size, is_training):

    name2feat = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'logit': tf.FixedLenFeature([n_label], tf.float32),
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

def create_model(flags, embed_tb, input_ids, logit, labels, n_label, is_training):
    
    stud_model = model.Student(
        flags=flags,
        embed_tb=embed_tb,
        input_ids=input_ids
    )

    output = stud_model.get_output()

    with tf.variable_scope("student/loss", reuse=tf.AUTO_REUSE):
        if is_training:
            output = tf.nn.dropout(output, rate=flags.dropout)

        logits = tf.keras.layers.Dense(n_label)(output)
        log_prob = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=n_label, dtype=tf.float32)
        nll = -tf.reduce_sum(one_hot_labels*log_prob, axis=-1)
        mse = tf.reduce_sum(tf.squared_difference(logit, logits), axis=-1)
        loss = tf.reduce_mean(nll + mse)

    return loss, log_prob

def model_fn_builder(flags, embed_tb):
    
    def model_fn(features, labels, mode, params):

        input_ids = features['input_ids']
        logit = features['logit']
        label = features['label_ids']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, log_prob = create_model(
            flags=flags,
            embed_tb=embed_tb,
            input_ids=input_ids,
            logit=logit,
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
            train_op = tf.train.AdamOptimizer(flags.learning_rate).minimize(loss, tf.train.get_global_step())
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
                predictions={'pred': pred}
            )

        return output_spec

    return model_fn

def main(FLAG):

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAG.output_dir)

    if FLAG.do_train:
        train_record_path = os.path.join(FLAG.train_data_path)
        train_input_fn = tfrecord_input_fn_builder(
            input_file=train_record_path,
            seq_length=FLAG.max_seq_length,
            n_label=FLAG.n_label,
            batch_size=FLAG.train_batch_size,
            is_training=True
        )
    if FLAG.do_valid:
        dev_record_path = os.path.join(FLAG.valid_data_path)
        dev_input_fn = tfrecord_input_fn_builder(
            input_file=dev_record_path,
            seq_length=FLAG.max_seq_length,
            n_label=FLAG.n_label,
            batch_size=FLAG.dev_batch_size,
            is_training=False
        )
    if FLAG.do_test:
        test_record_path = os.path.join(FLAG.test_data_path)
        test_input_fn = tfrecord_input_fn_builder(
            input_file=test_record_path,
            seq_length=FLAG.max_seq_length,
            n_label=FLAG.n_label,
            batch_size=FLAG.test_batch_size,
            is_training=False
        )
    
    embed_tb = np.load(FLAG.embed_tb_path)
    model_fn = model_fn_builder(FLAG, embed_tb)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAG.output_dir,
        save_summary_steps=FLAG.save_summary_steps,
        save_checkpoints_steps=FLAG.save_checkpoints_steps,
        keep_checkpoint_max=FLAG.keep_checkpoint_max,
        log_step_count_steps=FLAG.log_step_count_steps
    )
    warm_config = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAG.ckpt_path,
        vars_to_warm_start='student/*'
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
        tf.logging.info('  Train batch size = {}'.format(FLAG.train_batch_size))
        tf.logging.info('  Eval batch size = {}'.format(FLAG.dev_batch_size))
        tf.logging.info('  Num steps = {}'.format(FLAG.n_train_step))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAG.n_train_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, start_delay_secs=0, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    if FLAG.do_test:
        tf.logging.info('***************************')
        tf.logging.info('***** Running Testing *****')
        tf.logging.info('***************************')
        tf.logging.info('  Batch size = {}'.format(FLAG.test_batch_size))
