#!/usr/bin/env python  

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float32MultiArray


def Autoencoder(x, layer=[], act=tf.sigmoid):
    # layer[0] : input layer
    # x : input vector

    layer2 = layer[:-1]
    layer2.reverse()

    layer_num = len(layer)
    layer2_num = len(layer2)
    model = [x]
    idx = 0
    
    with tf.variable_scope("encoder"):
        for i in range(1,layer_num):
            model.append( tf.contrib.layers.fully_connected(model[idx], layer[i], activation_fn=act, 
                                                                                  scope="layer{0}".format(i),
                                                                                  weights_regularizer = tf.contrib.layers.l2_regularizer(0.001)) )
            idx = idx+1
    
    with tf.variable_scope("decoder"):
        for i in range(0,layer2_num-1):
            model.append( tf.contrib.layers.fully_connected(model[idx], layer2[i], activation_fn=act, 
                                                                                   scope="layer{0}".format(i+layer_num),
                                                                                   weights_regularizer = tf.contrib.layers.l2_regularizer(0.001)) )
            idx = idx+1
        
        model.append( tf.contrib.layers.fully_connected(model[idx], layer2[layer2_num-1], activation_fn=None, 
                                                                                          scope="layer{0}".format(layer2_num+layer_num-1),
                                                                                          weights_regularizer = tf.contrib.layers.l2_regularizer(0.001)) )


    Embeding = model[layer_num-1]
    Reconstruct = model[-1]

    return Embeding, Reconstruct



if __name__ == '__main__' :
    rospy.init_node('Autoencoder')
    rate = rospy.Rate(100)

    learning_rate_RMSProp = 0.01
    learning_rate_Gradient_Descent = 0.5

    training_epochs = 5000
    batch_size = 100
    log_display_step = 10
    examples_to_show = 10

    n_input = 5
    n_hidden1 = 4
    n_hidden2 = 3
    n_hidden3 = 2

    # Stacked Autoencoder
    X = tf.placeholder(tf.float32,[None,n_input])
    Y_true = X

    with tf.variable_scope("encoder"):
        
        hidden1 = tf.contrib.layers.fully_connected(X, n_hidden1, activation_fn=tf.sigmoid, scope="hidden1")
        hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, activation_fn=tf.sigmoid, scope="hidden2")
        hidden3 = tf.contrib.layers.fully_connected(hidden2, n_hidden3, activation_fn=tf.sigmoid, scope="hidden3")

    with tf.variable_scope("decoder"):

        hidden4 = tf.contrib.layers.fully_connected(hidden3, n_hidden2, activation_fn=tf.sigmoid, scope="hidden4")
        hidden5 = tf.contrib.layers.fully_connected(hidden4, n_hidden1, activation_fn=tf.sigmoid, scope="hidden5")
        Y_out = tf.contrib.layers.fully_connected(hidden5, n_input, activation_fn=tf.sigmoid, scope="Y_out")

        

    # Optimizer op
    loss = tf.reduce_mean(tf.pow(Y_true - Y_out,2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate_RMSProp)
    train = optimizer.minimize(loss)

    # Make filename queue & Reader
    filename_queue = tf.train.string_input_producer(["tf_data/testData_test.csv"],shuffle=False, name='filename_queue')
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Make csv read & parsing op 
    record_defaults = [[1.] for _ in range(n_input)]
    csv_data = tf.decode_csv(value, record_defaults=record_defaults)
    get_batch = tf.train.batch([csv_data],batch_size=batch_size)

    init = tf.initialize_all_variables()


          

    sess = tf.Session();

    sess.run(init)

    # Start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0

    with coord.stop_on_exception():
        while not rospy.is_shutdown() and not coord.should_stop():
            step = step+1

            if step == training_epochs :
                break;

            X_batch = sess.run(get_batch)

            _, loss_ = sess.run([train, loss], feed_dict={ X: X_batch })

            if step % log_display_step == 0 :
                print "step : {0}     loss : {1}".format(step,loss_)

            rate.sleep() 

    coord.request_stop()
    coord.join(threads)


