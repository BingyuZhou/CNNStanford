"""
GoogleNet image recognition

Author: Bingyu Zhou
"""

import tensorflow as tf
import tensorboard as tb
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import get_CIFAR10_data


def inception_v2(input, depth):
    """ inception layer1 """
    d1, d2_1, d2_2, d3_1, d3_2, d4 = depth

    conv1 = tf.layers.conv2d(
        input, filters=d1, kernel_size=1, padding='same')
    bn1 = tf.layers.batch_normalization(conv1, axis=3)
    layer1 = tf.nn.relu(bn1)

    conv2_1 = tf.layers.conv2d(
        input, d2_1, 1, padding='same')
    bn2 = tf.layers.batch_normalization(conv2_1, axis=3)
    layer2_1 = tf.nn.relu(bn2)
    conv2_2 = tf.layers.conv2d(layer2_1, d2_2, 3, 1,
                               padding='same')
    bn3 = tf.layers.batch_normalization(conv2_2, axis=3)
    layer2 = tf.nn.relu(bn3)

    conv3_1 = tf.layers.conv2d(
        input, d3_1, 1, padding='same')
    bn4 = tf.layers.batch_normalization(conv3_1, axis=3)
    layer3_1 = tf.nn.relu(bn4)
    conv3_2 = tf.layers.conv2d(
        layer3_1, d3_2, 5, padding='same')
    bn5 = tf.layers.batch_normalization(conv3_2, axis=3)
    layer3 = tf.nn.relu(bn5)

    conv4_1 = tf.layers.max_pooling2d(input, 3, 1, 'same')
    bn6 = tf.layers.batch_normalization(conv4_1, axis=3)
    layer4_1 = tf.nn.relu(bn6)
    conv4_2 = tf.layers.conv2d(
        layer4_1, d4, 1, 1, padding='same')
    bn7 = tf.layers.batch_normalization(conv4_2, axis=3)
    layer4 = tf.nn.relu(bn7)

    output = tf.concat([layer1, layer2, layer3, layer4], axis=3)
    return output


def GoogleNet(features, labels, mode):
    "Googlenet model"
    X = features["X"]
    conv1 = tf.layers.conv2d(X, 64, 7, 2, 'same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 3, 2, 'same')

    conv2_1 = tf.layers.conv2d(pool1, 64, 1, 1, 'same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv2_1, 192, 3, 1, 'same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 3, 2, 'same')

    inception1 = inception_v2(pool2, (64, 96, 128, 16, 32, 32))
    inception2 = inception_v2(inception1, (128, 128, 192, 32, 96, 64))

    pool3 = tf.layers.max_pooling2d(inception2, 3, 2, 'same')

    inception3 = inception_v2(pool3, (192, 96, 208, 16, 48, 64))
    inception4 = inception_v2(inception3, (160, 112, 224, 24, 64, 64))
    inception5 = inception_v2(inception4, (128, 128, 256, 24, 64, 64))
    inception6 = inception_v2(inception5, (112, 144, 288, 32, 64, 64))
    inception7 = inception_v2(inception6, (256, 160, 320, 32, 128, 128))
    pool4 = tf.layers.max_pooling2d(inception7, 3, 2, 'same')

    inception8 = inception_v2(pool4, (256, 160, 320, 32, 128, 128))
    inception9 = inception_v2(inception8, (384, 192, 384, 48, 128, 128))
    pool5 = tf.layers.average_pooling2d(inception9, 7, 1, 'valid')

    drop = tf.layers.dropout(pool5, 0.4, training=mode ==
                             tf.estimator.ModeKeys.TRAIN)

    drop_flat = drop.reshape((-1, 1024))
    linear = tf.layers.dense(drop_flat, 1000)

    predictions = {"classes": tf.argmax(linear, axis=1),
                   "probabilities": tf.nn.softmax(linear, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), 1000)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, linear)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels, predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode, loss, eval_metric_ops)


def GoogleNet_CIFAR(features, labels, mode):
    "Googlenet model"
    X = features["X"]

    conv1 = tf.layers.conv2d(X, 32, 5, 1, 'same')
    bn1 = tf.layers.batch_normalization(conv1, axis=3)
    layer1 = tf.nn.relu(bn1)

    inception1 = inception_v2(layer1, (32, 32, 64, 16, 32, 32))

    pool1 = tf.layers.max_pooling2d(inception1, 3, 2, 'same')

    inception2 = inception_v2(pool1, (64, 64, 96, 16, 32, 64))
    pool2 = tf.layers.max_pooling2d(inception2, 3, 2, 'same')

    inception3 = inception_v2(pool2, (120, 32, 64, 32, 64, 64))
    pool3 = tf.layers.max_pooling2d(inception3, 3, 2, 'same')

    inception4 = inception_v2(pool3, (120, 64, 96, 32, 64, 64))
    pool3 = tf.layers.average_pooling2d(inception4, 4, 1, 'valid')

    drop = tf.layers.dropout(pool3, 0.35, training=mode ==
                             tf.estimator.ModeKeys.TRAIN)

    drop_flat = tf.reshape(drop, [-1, 344])
    linear = tf.layers.dense(drop_flat, units=10)

    predictions = {"classes": tf.argmax(linear, axis=1),
                   "probabilities": tf.nn.softmax(linear, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels, predictions["classes"])}

    # loss
    loss = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(labels, linear))
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(
            0.0015, global_step, 100, 0.95, True)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op, eval_metric_ops=eval_metric_ops)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir="/tmp/GoogleNet")

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main():
    tf.reset_default_graph()
    # load data
    data = get_CIFAR10_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    for k, v in data.items():
        print('%s: ' % k, v.shape)
    # create estimator
    est = tf.estimator.Estimator(
        model_fn=GoogleNet_CIFAR, model_dir="/tmp/GoogleNet")

    # set up looing hook
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors_to_log, every_n_iter=100)

    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train}, y=y_train, batch_size=32, num_epochs=30, shuffle=True)
    est.train(input_fn=train_input_fn, hooks=[logging_hook], steps=None)

    # evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_val}, y=y_val, num_epochs=10, shuffle=False)
    eval_results = est.evaluate(eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    main()
