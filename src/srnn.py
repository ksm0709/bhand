#!/usr/bin/env python  
import sys
import time
import argparse
import math

import tensorflow as tf
import numpy as np
import threading
import Queue

import rospy
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Int8
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

from DRNNCell_impl import DRNNCell

MODE_PREDICTION = 0
MODE_LEARNING = 1


def weight_variable(shape):
    initial = tf.truncated_normal( shape, stddev=0.1 )
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name='biases')

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)

        if act == None :
            activations = preactivate
        else :
            activations = act(preactivate, 'activation')
        
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations

def inference(xu_cur):

    hidden1 = nn_layer(xu_cur, self.layer_in, self.layer_h, 'hidden1', act=tf.nn.relu)
    hidden2 = nn_layer(hidden1, self.layer_h, self.layer_h , 'hidden2', act=tf.nn.relu)
    
    x_next = nn_layer(hidden2, self.layer_h, self.layer_out, 'x(k+1)', act=None)
    
#    sfm = tf.nn.softmax(fout)

    keep_prob = tf.constant(1.0)
    x_next_drop= tf.nn.dropout(fout,keep_prob=keep_prob) 
    
    return x_next_drop 

def loss_function(fout, freal):
    #loss_val = tf.losses.mean_squared_error(freal,fout,weights=1.0)
    loss_val = tf.losses.softmax_cross_entropy(freal,fout,weights=1.0)

    return tf.reduce_mean(loss_val)

def training(loss, learning_rate):

    tf.summary.scalar('loss',loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    global_step = tf.Variable(0,name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def Cald_projected_gradient(grad):

    return grad


class RosTF():
    def __init__(self):
      
        ###Common Vars
        self.emgSampled = False
        self.stateSampled = False
    
        ###Network Vars(Hyper Parameters)
        self.size_x = 4
        self.size_u = 2
        self.layer_in = self.size_x + self.size_u
        self.layer_out = self.size_x
        self.layer_h = 8 
        self.batch_size = 20 
        self.learning_rate = 0.001
        self.seq_length = 5
            
        ###ROS Init
        self.subEmg = rospy.Subscriber('/actEMG',
                Float32MultiArray,
                self.callbackEmg, 
                queue_size=1)

        self.subState = rospy.Subscriber('/rrbot/joint_states',
                JointState,
                self.callbackState, 
                queue_size=1)

        self.pubPredict = rospy.Publisher('/srnn/Prediction',
                Float32MultiArray,
                queue_size=1)

            
        ###Tensorflow Init 
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)


        #Build Graph
        self.graph = tf.Graph()

        with self.graph.as_default():
        
            #Session
            self.sess = tf.Session()
        
            with self.sess.as_default():
          
                #Placeholders
                self.u_ph = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.size_u])
                self.x_ph = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.layer_out])
                self.uEnque_ph = tf.placeholder(tf.float32, shape=[self.seq_length, self.size_u])
                self.xEnque_ph = tf.placeholder(tf.float32, shape=[self.seq_length, self.size_x])
                self.x0Enque_ph = tf.placeholder(tf.float32, shape=[self.size_x])
                self.initial_state = tf.placeholder(tf.float32, shape=[None,self.layer_out])

                #DRNNNetwork
                self.cell = DRNNCell(num_output=self.layer_out, num_units=[10,10,10], activation=tf.nn.relu) 
                self.x_next, _states = tf.nn.dynamic_rnn( self.cell, self.u_ph, initial_state=self.initial_state, dtype=tf.float32 )
                self.W1 = get_W1()
                self.W2 = get_W2()
                self.b = get_b()

                self.loss = tf.reduce_sum(tf.square(self.x_ph - self.x_next)) 
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.get_gradient = self.optimizer.compute_gradients(self.loss)
                self.modify_gradient = calc_projected_gradient(self.get_gradient)
                self.set_gradient = slef.optimizer.apply_gradients(self.modify_gradient)
                self.train_op = self.optimizer.minimize(self.loss) 

                tf.summary.scalar( 'loss' , self.loss )

                #SRNNNetwork
                #self.x_next = inference( self.xu_ph ) 
                #self.loss = loss_function( self.x_next, self.x_ph )
                #self.train_op = training( self.loss, self.learning_rate ) 
                   
                #Summary
                self.summary = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

                #Init Variable
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)
                self.saver = tf.train.Saver()

                # Load Model ################################################# 
                self.model_dir = "tf_model/model_srnn.ckpt"
                self.saver_on = args.save
                self.loader_on = args.load 

                if self.loader_on == True:
                    self.saver.restore(self.sess,self.model_dir)
               ############################################################### 

                #Queue & Coordinator
                self.coord = tf.train.Coordinator()
                self.que = tf.FIFOQueue(shapes=[[self.seq_length,self.size_u],[self.seq_length,self.size_x],[self.size_x]], # u, x, x0
                                        dtypes=[tf.float32, tf.float32, tf.float32],
                                        capacity=1000)
                #self.que = tf.RandomShuffleQueue(shapes=[[self.layer_in],[self.layer_out]],
                #                                 dtypes=[tf.float32, tf.float32],
                #                                 capacity=100, min_after_dequeue=20, seed=2121)


                self.queClose_op = self.que.close(cancel_pending_enqueues=True)
                self.enque_op = self.que.enqueue([self.uEnque_ph, self.xEnque_ph, self.x0Enque_ph])
                self.deque_op = self.que.dequeue_many(self.batch_size) 
                self.que_size = self.que.size()

                #ETC
                self.f4ph = tf.placeholder( tf.float32, shape=[ self.size_x ])
                self.l2norm = tf.norm( self.f4ph, ord=2 )
                
                print "-------------------------------------------------"

    def callbackEmg(self, msg): 
        self.emgData = msg.data
        self.emgSampled = True

    def callbackState(self, msg):
        self.stateData = msg
        self.stateSampled = True
    
    def enque_thread(self):
        print "enque_thread : start"

        x_buf = []
        u_buf = []

        with self.coord.stop_on_exception():
            # Make sequence buffer 
            while True:
                if self.stateSampled == True :
                    
                    x_buf.append(self.stateData.position + self.stateData.velocity)
                    u_buf.append(self.stateData.effort)

                    self.emgSampled = False
                    self.stateSampled = False

                    if len(u_buf) == self.seq_length :
                        break

            # Enqueue
            while not self.coord.should_stop():
                #if self.emgSampled == True and self.stateSampled == True:
                if self.stateSampled == True :
                    
                    x_buf.append(self.stateData.position + self.stateData.velocity)
                    u_buf.append(self.stateData.effort)

                    x0 = np.array( x_buf[0] )
                    x = np.array( x_buf[1:self.seq_length+1] )
                    u = np.array( u_buf[0:self.seq_length] )

                    x_buf.pop(0)
                    u_buf.pop(0) 

                    self.sess.run(self.enque_op, feed_dict={ self.uEnque_ph : u,
                                                             self.xEnque_ph : x,
                                                             self.x0Enque_ph : x0} )

                    self.emgSampled = False
                    self.stateSampled = False

                time.sleep(0.001)

    def train_thread(self):
        print "train_thread : start"

        with self.coord.stop_on_exception():

            step = 0
            while not self.coord.should_stop():

                size = self.sess.run( self.que_size )

                if size > self.batch_size : 
                    u, x, x0 = self.sess.run( self.deque_op )
                  
                    feed_dict = { self.u_ph : u, 
                                  self.x_ph : x, 
                                  self.initial_state : x0  } 

                    _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                    step = step+1

                    if step % 10 == 0 :

                        print "step : {0}   loss : {1}".format(step, loss)

                        summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
                        self.summary_writer.add_summary( summary_str, step )
                        self.summary_writer.flush()
                        
                        if step%100 == 0 and self.saver_on == True:
                            self.saver.save(self.sess, self.model_dir) 

                time.sleep(0.001)
           

    def predict_thread(self): 
        print "prediction_thread : start"

        with self.coord.stop_on_exception():


            step = 0
            x0 = np.array([])
            u = np.array([])
            while not self.coord.should_stop():
                if self.stateSampled == True:
                    self.stateSampled = False
                   
                    x0 = np.array( [self.stateData.position + self.stateData.velocity] )
                    u = np.array( [[self.stateData.effort for _ in range(self.seq_length)]] )

                    feed_dict = { self.u_ph : u, 
                                  self.initial_state : x0  } 

                    result = self.sess.run(self.x_next, feed_dict=feed_dict)

                    step = step + 1

                    if step%10 == 0 and step > 0 :
                        
                        err = self.x_predict - x0[0,:]
                        l2err = self.sess.run(self.l2norm, feed_dict={ self.f4ph : err })
                        print "L2(ERR) : {0}   Prediction Error : {1}\n".format(l2err,err)

                    self.x_predict = result[0,0,:]

                time.sleep(0.001)


    def main(self):

        threads = [ threading.Thread(target=self.enque_thread) ,
                    threading.Thread(target=self.train_thread) ,
                    threading.Thread(target=self.predict_thread) ]    

        self.coord.register_thread( threads[0] )
        self.coord.register_thread( threads[1] )
        self.coord.register_thread( threads[2] )

        
        # Train thread or Predict thread start
        if args.mode == MODE_LEARNING:
            threads[0].start() #Enque thread
            threads[1].start() #Train thread
        
            rospy.init_node('Estimator')
            rate = rospy.Rate(100)            

            with self.coord.stop_on_exception():
                while not rospy.is_shutdown() and not self.coord.should_stop():
                    rate.sleep() 
        
        else : 
            threads[2].start() #Predict thread
           
            rospy.init_node('Estimator')
            rate = rospy.Rate(100)            

            with self.coord.stop_on_exception():
                while not rospy.is_shutdown():
                    rate.sleep() 
        

        self.sess.run( self.queClose_op )
        self.coord.request_stop()
        self.coord.join(threads) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '-mode',
            type=int,
            default=0,
            help="Mode configuration [ 0: prediction only(default) , 1: prediction with learning ]"
            )
   
    parser.add_argument(
            '-log_dir',
            type=str,
            default='tf_log',
            help="Logging directory ( default : tf_log )"
            )

    parser.add_argument(
            '-load',
            type=bool,
            default=False,
            help="Load model"
            )

    parser.add_argument(
            '-save',
            type=bool,
            default=True,
            help="Save model"
            )
    

    args, unparsed = parser.parse_known_args()

    tensor = RosTF()
    tensor.main()
