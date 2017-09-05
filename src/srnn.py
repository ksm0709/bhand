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
from autoencoder import Autoencoder

MODE_PREDICTION = 3
MODE_TRAIN_ALL = 0
MODE_TRAIN_AE = 1
MODE_TRAIN_DRNN = 2

def calc_projected_gradient(grad):

    return grad

def set_optimizer(y_real,y_out,rate):
    loss = tf.reduce_sum(tf.square(y_real - y_out)) 
    optimizer = tf.train.AdamOptimizer(rate)
    train_op = optimizer.minimize(loss) 

    return train_op, loss

class RosTF():
    def __init__(self):
      
        ###Common Vars
        self.emgSampled = False
        self.stateSampled = False

        ###Network Vars(Hyper Parameters)
        self.size_zx = 3
        self.size_zu = 6
        self.size_x = 5
        self.size_u = 8
        self.layer_in = self.size_zx + self.size_zu
        self.layer_out = self.size_zx
        self.batch_size = 20 
        self.learning_rate = 0.01
        self.seq_length = 1
            
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
                self.zu_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_zu])
                self.zx_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_zx])
                self.u_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_u])
                self.x_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_x])
                self.uEnque_ph = tf.placeholder(tf.float32, shape=[None, self.size_u])
                self.xEnque_ph = tf.placeholder(tf.float32, shape=[None, self.size_x])
                self.x0Enque_ph = tf.placeholder(tf.float32, shape=[self.size_x])
                self.initial_state = tf.placeholder(tf.float32, shape=[None,self.size_zx])

                #Autoencoder
                with tf.variable_scope("AE_u"):
                    self.zu, self.ru = Autoencoder(self.u_ph, range(self.size_u,self.size_zu-1,-1))
                with tf.variable_scope("AE_x"):
                    self.zx, self.rx = Autoencoder(self.x_ph, range(self.size_x,self.size_zx-1,-1))

                #DRNNNetwork
                self.cell = DRNNCell(num_output=self.layer_out, num_units=[40,30], activation=tf.nn.relu) 
                self.zx_next, _states = tf.nn.dynamic_rnn( self.cell, self.zu_ph, initial_state=self.initial_state, dtype=tf.float32 )

                #Optimizer
                self.train_drnn, self.loss_drnn = set_optimizer(self.zx_ph, self.zx_next, self.learning_rate)
                self.train_zu, self.loss_zu = set_optimizer(self.u_ph, self.ru, self.learning_rate)
                self.train_zx, self.loss_zx = set_optimizer(self.x_ph, self.rx, self.learning_rate)
                self.train_func = {
                    MODE_TRAIN_AE : self.train_ae,
                    MODE_TRAIN_DRNN : self.train_drnn,
                    MODE_TRAIN_ALL : self.train_all,
                    }

                tf.summary.scalar( 'drnn_loss' , self.loss_drnn )
                tf.summary.scalar( 'ae_u_loss' , self.loss_zu )
                tf.summary.scalar( 'ae_x_loss' , self.loss_zx )

                
                #Init Variable
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)
                self.saver = tf.train.Saver()

                # Load Model ################################################# 
                self.model_dir = "~/catkin_ws/src/bhand/src/tf_model/model_srnn.ckpt"
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
                
                #Summary
                self.summary = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

                print "-------------------------------------------------"
    
    

    def train_all(self,step,feed_dict):
        _,_,_,loss_zx,loss_zu,loss_drnn = self.sess.run([self.train_zx, 
                                                         self.train_zu, 
                                                         self.train_drnn, 
                                                         self.loss_zx, 
                                                         self.loss_zu, 
                                                         self.loss_drnn], 
                                                         feed_dict=feed_dict)
                                                            
        if step % 10 == 0:
           print "step : {0}   loss_AE_x : {1}    loss_AE_u : {2}    loss_drnn : {3}".format(step,loss_zx, loss_zu, loss_drnn)

           summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
           self.summary_writer.add_summary( summary_str, step )
           self.summary_writer.flush()
            
           if step%100 == 0 and self.saver_on == True:
               self.saver.save(self.sess, self.model_dir)

        return 0

    def train_ae(self,step,feed_dict):
        _,_,loss_zx,loss_zu = self.sess.run([self.train_zx, 
                                             self.train_zu, 
                                             self.loss_zx, 
                                             self.loss_zu], feed_dict=feed_dict)

        if step % 10 == 0 :

            print "step : {0}   loss_AE_x : {1}    loss_AE_u : {2}".format(step,loss_zx, loss_zu)

            summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
            self.summary_writer.add_summary( summary_str, step )
            self.summary_writer.flush()
            
            if step%100 == 0 and self.saver_on == True:
                self.saver.save(self.sess, self.model_dir)

        return 0
    def train_drnn(self,step,feed_dict):

        _,loss_drnn = self.sess.run([self.train_drnn, self.loss_drnn], feed_dict=feed_dict)

        if step % 10 == 0 :

            print "step : {0}   loss_drnn : {1}".format(step, loss_drnn)

            summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
            self.summary_writer.add_summary( summary_str, step )
            self.summary_writer.flush()
            
            if step%100 == 0 and self.saver_on == True:
                self.saver.save(self.sess, self.model_dir)

        return 0

    def callbackEmg(self, msg): 
        self.emgData = msg.data
        self.emgSampled = True

    def callbackState(self, msg):
        self.stateData = msg.data
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

                    x0 = np.reshape(x0, [self.batch_size,1,self.size_x])

                    # Encoding
                    zu = self.sess.run( self.zu, feed_dict={ self.u_ph : u })
                    zx = self.sess.run( self.zx , feed_dict={ self.x_ph : x })
                    zx0 = self.sess.run( self.zx , feed_dict={ self.x_ph : x0 })

                    zx0 = np.reshape(zx0, [self.batch_size,self.size_zx])
                  
                    feed_dict = { self.u_ph : u,
                                  self.x_ph : x,
                                  self.zu_ph : zu, 
                                  self.zx_ph : zx, 
                                  self.initial_state : zx0  } 

                    step = step+1

                    # Training & Print & Saving model
                    self.train_func[args.mode](step,feed_dict)

                time.sleep(0.001)
           

    def predict_thread(self): 
        print "prediction_thread : start"

        with self.coord.stop_on_exception():

            step = 0
            x0 = np.array([])
            u = np.array([])
            while not self.coord.should_stop():
                if self.stateSampled == True and self.emgSampled == True:
                    self.stateSampled = False
                    self.emgSampled = False
                   
                    x0 = np.array( [[self.stateData]] )
                    u = np.array( [[self.emgData]] )

                    zx0 = self.sess.run( self.zx , feed_dict={ self.x_ph : x0 })
                    zu = self.sess.run( self.zu, feed_dict={ self.u_ph : u })

                    zx0 = np.reshape(zx0, [1,self.size_zx])

                    feed_dict = { self.zu_ph : zu, 
                                  self.initial_state : zx0  } 

                    result = self.sess.run(self.zx_next, feed_dict=feed_dict)

                    step = step + 1

                    if step%10 == 0 and step > 0 :
                        
                        rx0 = self.sess.run(self.rx, feed_dict={ self.x_ph : x0 })
                        ru = self.sess.run(self.ru, feed_dict={ self.u_ph : u})
                        
                        err_drnn = self.x_predict - zx0[0,:]
                        err_x0 = x0[0,:] - rx0[0,:]
                        err_u = u[0,0,:] - ru[0,0,:]

                        l2err_drnn = np.linalg.norm(err_drnn,2)
                        l2err_x0 = np.linalg.norm(err_x0,2)
                        l2err_u = np.linalg.norm(err_u,2) 

                        print " [ Estimation erros ]\nDRNN : {0}\nAE_x : {1}\nAE_u : {2}".format(l2err_drnn, l2err_x0, l2err_u)

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
        if args.mode == MODE_PREDICTION:
            threads[2].start() #Predict thread
        
            rospy.init_node('Estimator')
            rate = rospy.Rate(100)            

            with self.coord.stop_on_exception():
                while not rospy.is_shutdown() and not self.coord.should_stop():
                    rate.sleep() 
        
        else : 
            threads[0].start() #Enque thread
            threads[1].start() #Train thread
            
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
            help="Mode configuration [ 0: prediction only(default) , 1: train all, 2: train AE, 3: train DRNN ]"
            )
   
    parser.add_argument(
            '-log_dir',
            type=str,
            default='~/catkin_ws/src/bhand/src/tf_log',
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
