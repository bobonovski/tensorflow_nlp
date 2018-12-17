#-*- coding:utf-8 -*- 
"""Recurrent Convolutional Neural Networks for Text Classiﬁcation"""
import time
import os
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import data_helpers

class RCNN():
    def __init__(self, sentence_length, num_classes, vocab_size, 
            embedding_size, hidden_state_size):
        self.input_batch = tf.placeholder(tf.int32, [None, sentence_length], name="input_batch")
        self.output_batch = tf.placeholder(tf.float32, [None, num_classes], name="output_batch")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.words_embedding = tf.nn.embedding_lookup(self.W, self.input_batch)
            self.words_embedding_reverse = tf.reverse_sequence(self.words_embedding, 
                    seq_lengths=self.sequence_length, seq_dim=1)
        
        with tf.name_scope("bidirectional-rnn"):
            fw_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_state_size)
            bw_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_state_size)
            
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, 
                    self.words_embedding, sequence_length=self.sequence_length, dtype=tf.float32)
            rnn_outputs_fw, rnn_outputs_bw = rnn_outputs

            # get left context
            left_context = rnn_outputs_fw[:, :-1, :]
            left_context_padded = tf.pad(left_context, [[0, 0], [1, 0], [0, 0]])
            
            # get right context
            right_context = tf.reverse_sequence(rnn_outputs_bw, 
                    seq_lengths=self.sequence_length, seq_dim=1)[:, 1:, :]
            right_context_padded = tf.pad(right_context, [[0, 0], [0, 1], [0, 0]])
            
            # concat context features
            self.context_features = tf.concat([left_context_padded, 
                self.words_embedding, right_context_padded], axis=2)

        self.context_features_expanded = tf.expand_dims(self.context_features, -1)
        context_features_shape = self.context_features.get_shape().as_list()
        
        with tf.name_scope("max-pool"):
            self.max_pooled_features = tf.nn.max_pool(self.context_features_expanded,
                    ksize=[1, context_features_shape[1], 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID', name="max-pool")

        self.flat_features = tf.reshape(self.max_pooled_features, [-1, context_features_shape[2]])

        # dropout layer
        with tf.name_scope("dropout"):
            self.flat_features_dropout = tf.nn.dropout(self.flat_features, self.dropout_keep_prob)
        
        # output layer
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([context_features_shape[2], num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.flat_features_dropout, W, b, name="output")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # loss layer
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.output_batch)
            self.loss = tf.reduce_mean(losses)

        # accuracy layer
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.output_batch, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

## program flags
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_state_size", 100, "RNN hidden state dimension (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# config parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# load data
documents, y, sequence_length = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# build vocabulary
max_doc_length = max([len(doc.split(" ")) for doc in documents])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length)
x = np.array(list(vocab_processor.fit_transform(documents, max_doc_length)))

# shuffle data
np.random.seed(9991)
shuffled_idx = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffled_idx]
y_shuffled = y[shuffled_idx]
sequence_length_shuffled = sequence_length[shuffled_idx]

# split train and test data
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
sequence_length_train, sequence_length_dev = sequence_length_shuffled[:dev_sample_index], sequence_length_shuffled[dev_sample_index:]

# training steps
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = RCNN(
            sentence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            hidden_state_size=FLAGS.hidden_state_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, sequence_length_batch):
            feed_dict = {
              model.input_batch: x_batch,
              model.output_batch: y_batch,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob,
              model.sequence_length: sequence_length_batch,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, sequence_length_batch, writer=None):
            feed_dict = {
              model.input_batch: x_batch,
              model.output_batch: y_batch,
              model.dropout_keep_prob: 1.0,
              model.sequence_length: sequence_length_batch,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), 
                sequence_length_train, FLAGS.batch_size, FLAGS.num_epochs)
        
        # Training loop. For each batch...
        for batch, sequence_length in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, sequence_length)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, sequence_length_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path)) 
