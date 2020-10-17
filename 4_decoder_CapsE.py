import torch
import os
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from corpus import *
from models_CapsE import CapsE

np.random.seed(1234)
tf.set_random_seed(1234)

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import numpy as np
from utils import save_model, load_object, load_model, save_object
import random
import time
from config import Config

args = Config()
args.load_config()
device = "cuda" if args.cuda else "cpu"





with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        capse = CapsE(sequence_length=x_valid.shape[1],
                            initialization=initialization,
                            embedding_size=args.embedding_dim,
                            filter_size=args.filter_size,
                            num_filters=args.num_filters,
                            vocab_size=len(words_indexes),
                            iter_routing=args.iter_routing,
                            batch_size=2*args.batch_size,
                            num_outputs_secondCaps=args.num_outputs_secondCaps,
                            vec_len_secondCaps=args.vec_len_secondCaps,
                            useConstantInit=args.useConstantInit
                            )

        # Define Training procedure
        #optimizer = tf.contrib.opt.NadamOptimizer(1e-3)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(capse.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(args.run_folder, "runs_CapsE", args.model_name))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                capse.input_x: x_batch,
                capse.input_y: y_batch
            }
            _, step, loss = sess.run([train_op, global_step, capse.total_loss], feed_dict)
            return loss

        num_batches_per_epoch = int((data_size - 1) / args.batch_size) + 1
        for epoch in range(args.num_epochs):
            for batch_num in range(num_batches_per_epoch):
                x_batch, y_batch = train_batch()
                loss = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                #print(loss)
            if epoch > 0:
                if epoch % args.savedEpochs == 0:
                    path = capse.saver.save(sess, checkpoint_prefix, global_step=epoch)
                    print("Saved model checkpoint to {}\n".format(path))




