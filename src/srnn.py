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

def Cald_projected_gradient(grad):

    return grad


class RosTF():
    def __init__(self):
      
        ###Common Vars
        self.emgSampled = False
        self.stateSampled = False
    
        ###Network Vars(Hyper Parameters)
        self.size_x = 20
        self.size_u = 8
        self.layer_in = self.size_x + self.size_u
        self.layer_out = self.size_x
        self.batch_size = 20 
        self.learning_rate = 0.001
        self.seq_length = 5
            
        ###ROS Init
        self.subEmg = rospy.Subscriber('/actEMG',
                Float32MultiArray,
                self.callbackEmg, 
                queue_size=1)

        self.subState = rospy.Subscriber('/finger_state',
                Float32MultiArray,
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
                self.cell = DRNNCell(num_output=self.layer_out, num_units=[40,30], activation=tf.nn.relu) 
                self.x_next, _states = tf.nn.dynamic_rnn( self.cell, self.u_ph, initial_state=self.initial_state, dtype=tf.float32 )
                
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
                if self.stateSampled == True and self.emgSampled == True :
                    
                    x_buf.append(self.stateData)
                    u_buf.append(self.emgData)

                    self.emgSampled = False
                    self.stateSampled = False

                    if len(u_buf) == self.seq_length :
                        break

            # Enqueue
            while not self.coord.should_stop():
                if self.emgSampled == True and self.stateSampled == True:
                    
                    x_buf.append(self.stateData)
                    u_buf.append(self.emgData)

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
                   
                    x0 = np.array( [self.stateData] )
                    u = np.array( [[self.emgData for _ in range(self.seq_length)]] )

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
            default=False,
            help="Save model"
            )
    

    args, unparsed = parser.parse_known_args()

    tensor = RosTF()
    tensor.main()
