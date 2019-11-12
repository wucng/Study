#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# This file contains functions for training a TensorFlow model
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

def process_dataset():
    # Import the data
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[28,28, 1]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_weight_npz(sess,filename='./models/tf_args.npz'):
    all_weights=tf.trainable_variables()
    # or
    # all_weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # sess = tf.keras.backend.get_session()
    tf_arg={}
    for weight in all_weights:
        name=weight.name
        # w=tf.get_default_graph().get_tensor_by_name(name)
        w=weight
        tf_arg[name]=sess.run(w)

    np.savez(filename,**tf_arg)

def save_weight_npz2(sess,filename='./models/tf_args.npz'):
    """将权重保存到文件中，便于后续使用tensorrt自行加载"""
    tf_args={}
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        tf_args[i.name]=sess.run(i)

    np.savez(filename, **tf_arg)

def save(model, filename):
    '''
    网络参数冻结，保存成.pb文件，用于tensorrt
    :param model:
    :param filename:
    :return:
    ''' # model.input.op.name # 'input_1'
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name # 'dense_1/Softmax'
    if tf.__version__<"2.0.0":
        sess = tf.keras.backend.get_session()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    else:
        sess = tf.compat.v1.keras.backend.get_session()
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),[output_names])
        frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

    # save weight to .npz
    save_weight_npz(sess)

def main():
    best_model_weights = './best_weights.h5'
    checkpoint = ModelCheckpoint(
        best_model_weights,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        period=1
    )
    earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='auto'
    )

    reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=40,
        verbose=1,
        mode='auto',
        cooldown=1
    )
    callbacks = [checkpoint, reduce, earlystop]

    x_train, y_train, x_test, y_test = process_dataset()
    model = create_model()
    # Train the model on the data
    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs = 1, verbose = 1,
              callbacks=callbacks)
    # Evaluate the model on test data
    model.evaluate(x_test, y_test)
    filename = "models/lenet5.pb"
    if not os.path.exists(os.path.dirname(filename)):os.makedirs(os.path.dirname(filename))
    save(model, filename=filename)
    '''
    model.save("mnist.h5") # 保存模型结构和参数
    model=tf.keras.models.load_model("mnist.h5")
    '''
if __name__ == '__main__':
    main()