#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import random
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import rospy
from std_msgs.msg import Int8
from std_msgs.msg import Float32MultiArray


# EMG data from ROS
emgData = None
sampled = False

def emg_callback(msg):
    global sampled, emgData
    emgTmp = np.array(msg.data)
    emgData = np.reshape(emgTmp, [1,16])
    sampled = True

# main function
if __name__ == '__main__':
    
    global sampled, emgData

    #option
    load_=False
    train_=True
    


    #load=False, train=True:    train with csv (firsttime)
    #load=True, train=True:     train with csv (overlap)
    #load=True, train=False :   predict with ros (saved model)

    seed = 7
    np.random.seed(seed)  # reproducibility

    # parameters
    learning_rate = 0.01
    training_epochs = 150
    batch_size = 5

    nb_classes = 7

    # input place holders
    X = tf.placeholder(tf.float32, [None, 16])
    Y = tf.placeholder(tf.int32, [None, 1])

    #drop-out
    keep_prob = tf.placeholder(tf.float32)

    #one-hot encoding 
    Y_one_hot = tf.one_hot(Y,nb_classes)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

    # weights & bias for nn layers 
    W1 = tf.Variable(tf.random_normal([16, 100]))
    b1 = tf.Variable(tf.random_normal([100]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([100, 100]))
    b2 = tf.Variable(tf.random_normal([100]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.Variable(tf.random_normal([100, 100]))
    b3 = tf.Variable(tf.random_normal([100]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    W4 = tf.Variable(tf.random_normal([100, 100]))
    b4 = tf.Variable(tf.random_normal([100]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    W5 = tf.Variable(tf.random_normal([100, 100]))
    b5 = tf.Variable(tf.random_normal([100]))
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)

    W6 = tf.Variable(tf.random_normal([100, 100]))
    b6 = tf.Variable(tf.random_normal([100]))
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)

    W7 = tf.Variable(tf.random_normal([100, 100]))
    b7 = tf.Variable(tf.random_normal([100]))
    L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)

    W8 = tf.Variable(tf.random_normal([100, 100]))
    b8 = tf.Variable(tf.random_normal([100]))
    L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)

    W9 = tf.Variable(tf.random_normal([100,nb_classes]))
    b9 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(L8, W9) + b9

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y_one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #accuracy
    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # saver
    saver = tf.train.Saver()
    model_dir = "tf_model/model_posture3.ckpt"


# initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

# restore
    if load_ == True:
        saver.restore(sess, model_dir)

# training mode
    if train_ == True :
        #data load
        dataframe = pandas.read_csv("21000data.csv", header=None)
        dataset = dataframe.values
        x= dataset[:,0:-1].astype(float)
        y= dataset[:,[-1]]
        
        #labelencoder
        en = LabelEncoder()
        en.fit(y)
        y = en.transform(y)
        y=np.array([y])
        y=y.T

        print("y:{0}".format(y))
        #y = np_utils.to_categorical(y)

    # train my model
        for epoch in range(training_epochs):
            avg_cost = 0
        #total_batch = int(mnist.train.num_examples / batch_size)

        #for i in range(total_batch):
            for i in range(batch_size):
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feed_dict = {X: x, Y: y , keep_prob: 0.7}
                c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            #avg_cost += c / total_batch
                loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
        #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
                saver.save(sess, model_dir)
            print('Epoch:', '%04d' % (epoch + 1))
            print("loss: {:.3f}\tacc: {:.2%}".format(loss,acc))

   
        print('Learning Finished!')
        #print('Accuracy:', sess.run(accuracy, feed_dict={X: x, Y: y, keep_prob: 1}))

# save trained network
       # saver.save(sess, model_dir)
# inference mode
    else :
        rospy.init_node('Estimator')
        subemgdata = rospy.Subscriber("/actEMG", Float32MultiArray, emg_callback)
        pubPosture = rospy.Publisher('/posture', Int8, queue_size=10)
        rate = rospy.Rate(100)

        rospy.loginfo("Estimator node start!")
    
        while not rospy.is_shutdown() :
            if sampled == True :
                pred = sess.run(prediction, feed_dict={X: emgData})
                
                posture_msg = Int8()
                posture_msg.data = pred[0]
                pubPosture.publish(posture_msg)

                sampled = False
                print("pred:{0}\data:{0}\msg:{0}".format(pred,emgData,posture_msg))
            rate.sleep()

      #  for p, y in zip(pred, y.flatten())
       #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))







