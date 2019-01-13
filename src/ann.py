#!/usr/bin/env python
import serial
import sys
import numpy as np
import tensorflow as tf
import random
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import rospy
from std_msgs.msg import Int8
from std_msgs.msg import Float32MultiArray


class ann_ros(object):

    def __init__(self):
        super(ann_ros, self).__init__()

        # EMG data from ROS
        self.emgData = None
        self.sampled = False

        #option
        load_=False
        train_=False

        if len(sys.argv) > 0 :
            for cmd in sys.argv:

                if cmd == '-l':
                    load_ = True
            
                elif cmd == '-t':
                    train_ = True
        

        #load=False, train=True:    train with csv (firsttime)
        #load=True, train=True:     train with csv (overlap)
        #load=True, train=False :   predict with ros (saved model)

        seed = 7
        np.random.seed(seed)  # reproducibility

        # parameters
        learning_rate = 0.01
        training_epochs = 150
        train_num = 10
        batch_size = 32
        nb_classes = 7

        # input place holders
        X = tf.placeholder(tf.float32, [None, 16])
        Y = tf.placeholder(tf.int32, [None, 1])
        #batch_X, batch_Y = tf.train.batch([X,Y],batch_size=batch_size)

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

        WO = tf.Variable(tf.random_normal([100,nb_classes]))
        bO = tf.Variable(tf.random_normal([nb_classes]))
        hypothesis = tf.matmul(L3, WO) + bO

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
        model_dir = "tf_model/model_posture.ckpt"


    # initialize
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    # restore
        if load_ == True:
            saver.restore(sess, model_dir)

    # training mode
        if train_ == True :
            #data load
            dataframe = pandas.read_csv("~/catkin_ws/src/bhand/src/21000data.csv", header=None)
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
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)

        # train my model
            for epoch in range(training_epochs):
                avg_cost = 0
                avg_acc = 0
            #total_batch = int(mnist.train.num_examples / batch_size)

            #for i in range(total_batch):
                for i in range(train_num):
                    #train
                    feed_dict = {X: x_train, Y: y_train , keep_prob: 0.7}
                    loss, _= sess.run([cost, optimizer], feed_dict=feed_dict)

                    #test
                    feed_dict = {X: x_test, Y: y_test , keep_prob: 0.7}
                    acc = sess.run(accuracy, feed_dict=feed_dict)

                    avg_cost = avg_cost + loss
                    avg_acc = avg_acc + acc

                avg_cost = avg_cost / train_num
                avg_acc = avg_acc / train_num

                print('Epoch:', '%04d' % (epoch + 1))
                print("loss: {:.3f}\tacc: {:.2%}".format(avg_cost,avg_acc))

                saver.save(sess, model_dir)
            print('Learning Finished!')
            #print('Accuracy:', sess.run(accuracy, feed_dict={X: x, Y: y, keep_prob: 1}))

    # save trained network
            saver.save(sess, model_dir)

    # inference mode
        else :
            rospy.init_node('Estimator')
            subemgdata = rospy.Subscriber("/actEMG", Float32MultiArray, self.emg_callback)
            pubPosture = rospy.Publisher('/posture', Int8, queue_size=10)
            rate = rospy.Rate(100)

            #comm = serial.Serial('/dev/ttyUSB0')

            rospy.loginfo("Estimator node start!")
        
            while not rospy.is_shutdown() :
                if self.sampled == True :
                    pred = sess.run(prediction, feed_dict={X: self.emgData})
                    
                    posture_msg = Int8()
                    posture_msg.data = pred[0]
                    pubPosture.publish(posture_msg)

                    # Send Posture to hand controller	
                    # comm.write('{:}'.format(pred[0]))

                    self.sampled = False
                    print("pred:{0}\data:{0}\msg:{0}".format(pred,self.emgData,posture_msg))
                rate.sleep()

            #comm.close()

    def emg_callback(self,msg):
        emgTmp = np.array(msg.data)
        self.emgData = np.reshape(emgTmp, [1,16])
        self.sampled = True

if __name__=='__main__':
    ann = ann_ros()
