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

MODE_PREDICTION = 0
MODE_TRAIN_ALL = 1
MODE_TRAIN_AE = 2
MODE_TRAIN_DRNN = 3
MODE_SAVE_DATA = 4

SAVE_ALL = 1
SAVE_AE = 2
SAVE_DRNN = 3

LOAD_ALL = 1
LOAD_AE = 2
LOAD_DRNN = 3

ADAM_OPTIMIZER = 0
GD_OPTIMIZER = 1
ADAGRAD_OPTIMIZER = 2


def kl_divergence(p, q):
    return p*tf.log(p/q) + (1-p)*tf.log((1-p)/(1-q))

def set_optimizer(y_real,y_out,rate,optimizer=ADAM_OPTIMIZER,scope=None,add_loss=None):

    optimizer_dict = { ADAM_OPTIMIZER : tf.train.AdamOptimizer(rate),
                       GD_OPTIMIZER : tf.train.GradientDescentOptimizer(rate),
                       ADAGRAD_OPTIMIZER : tf.train.AdagradOptimizer(rate)}

    mse_loss = 10000*tf.reduce_mean(tf.square(y_real - y_out))
    reg_loss = 10*tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=scope)

    if add_loss == None:
        loss = tf.add_n([mse_loss] + reg_loss)

        optimizer_op = optimizer_dict[optimizer]
        train_op = optimizer_op.minimize(loss) 

        return train_op, [loss,mse_loss,tf.reduce_sum(reg_loss)]

    else:
        loss = tf.add_n([mse_loss] + reg_loss + [add_loss])

        optimizer_op = optimizer_dict[optimizer]
        train_op = optimizer_op.minimize(loss) 

        return train_op, [loss,mse_loss,tf.reduce_sum(reg_loss),add_loss]

class RosTF():
    def __init__(self):
      
        ###Common Vars
        self.emgSampled = False
        self.stateSampled = False
        self.filenameHeader = args.filename

        ###Network Vars(Hyper Parameters)
        self.size_x = 5
        self.size_u = 8
        self.layer_in = self.size_x + self.size_u
        self.layer_out = self.size_x
        self.batch_size = 20
        self.learning_rate = 0.001
        self.seq_length = 10
        self.sparsity_target = 0.3
        self.sparsity_weight = 10
            
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

        #Make log file for model check
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)


        #Build Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
        
            #Make Session
            self.sess = tf.Session()
            with self.sess.as_default():
          
                #Placeholders
                self.u_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_u])   # U (emg)
                self.x_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_x])   # X (state)
                self.uEnque_ph = tf.placeholder(tf.float32, shape=[None, self.size_u]) 
                self.xEnque_ph = tf.placeholder(tf.float32, shape=[None, self.size_x])
                self.x0Enque_ph = tf.placeholder(tf.float32, shape=[self.size_x])
                self.initial_state = tf.placeholder(tf.float32, shape=[None,self.size_x])
                self.keep_prob_ph = tf.placeholder(tf.float32)

                #DRNNNetwork
                with tf.variable_scope("DRNN"):
                    self.cell = DRNNCell(num_output=self.layer_out, num_units=[30, 30, 30, 30, 30, 30], activation=tf.nn.tanh, output_activation=tf.nn.tanh, keep_prob=self.keep_prob_ph) 
                    self.x_next, _states = tf.nn.dynamic_rnn( self.cell, self.u_ph, initial_state=self.initial_state, dtype=tf.float32 )

                    self.train_drnn, self.loss_drnn = set_optimizer(self.x_ph, self.x_next, self.learning_rate,scope="DRNN")
                    tf.summary.scalar( 'drnn_loss' , self.loss_drnn[0] )
                    tf.summary.scalar( 'drnn_loss_mse', self.loss_drnn[1])
                    tf.summary.scalar( 'drnn_loss_reg', self.loss_drnn[2])

                #Init Variable
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)
                
                # Saver / Load Model ######################################### 
                self.drnn_model_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_model/model_DRNN.ckpt"
                
                self.drnn_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DRNN"))

                if args.load == LOAD_ALL :
                    self.drnn_saver.restore(self.sess,self.drnn_model_dir)
                elif args.load == LOAD_DRNN:
                    self.drnn_saver.restore(self.sess,self.drnn_model_dir)

               ############################################################### 

                #Queue & Coordinator
                self.coord = tf.train.Coordinator()
                self.que = tf.FIFOQueue(shapes=[[self.seq_length,self.size_u],[self.seq_length,self.size_x],[self.size_x]], # u, x, x0
                                        dtypes=[tf.float32, tf.float32, tf.float32],
                                        capacity=1000, min_after_dequeue=500)
               # self.que = tf.RandomShuffleQueue(shapes=[[self.seq_length,self.size_u],[self.seq_length,self.size_x],[self.size_x]],
               #                                  dtypes=[tf.float32, tf.float32, tf.float32],
               #                                  capacity=3000, min_after_dequeue=2000, seed=2121)


                self.queClose_op = self.que.close(cancel_pending_enqueues=True)
                self.enque_op = self.que.enqueue([self.uEnque_ph, self.xEnque_ph, self.x0Enque_ph])
                self.deque_op = self.que.dequeue_many(self.batch_size) 
                self.que_size = self.que.size()

                #File input pipeline
                filename_queue_emg = tf.train.string_input_producer(["tf_data/emg_state/1_emg.csv"],shuffle=False, name='filename_queue_emg')
                filename_queue_state = tf.train.string_input_producer(["tf_data/emg_state/1_state.csv"],shuffle=False, name='filename_queue_state')

                reader_emg = tf.TextLineReader()
                reader_state = tf.TextLineReader()
                key_emg, value_emg = reader_emg.read(filename_queue_emg)
                key_state, value_state = reader_state.read(filename_queue_state)

                csv_data_emg = tf.decode_csv(value_emg, record_defaults=[[1.] for _ in range(self.size_u)])
                csv_data_state = tf.decode_csv(value_state, record_defaults=[[1.] for _ in range(self.size_x)])

                self.get_data_emg = tf.train.batch([csv_data_emg],batch_size=1)
                self.get_data_state = tf.train.batch([csv_data_state], batch_size=1)
                self.que_thread = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)


                #ETC
                self.f4ph = tf.placeholder( tf.float32, shape=[ self.size_x ])
                self.l2norm = tf.norm( self.f4ph, ord=2 )
                
                #Summary
                self.summary = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

                print "-------------------------------------------------"
    
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

                    if len(u_buf) == self.seq_length+1 :
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

                    x0 = np.reshape(x0, [self.batch_size,self.size_x])

                    feed_dict = { self.u_ph : u,
                                  self.x_ph : x,
                                  self.initial_state : x0,
                                  self.keep_prob_ph : 0.8
                                  } 
                    
                    # Training
                    _,loss_drnn = self.sess.run([self.train_drnn, self.loss_drnn], feed_dict=feed_dict)

                    if step % 10 == 0 :

                        print "step : {:d}   loss_drnn : {:5.5f}".format(step, loss_drnn[0])

                        summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
                        self.summary_writer.add_summary( summary_str, step )
                        self.summary_writer.flush()
            
                        if step%1000 == 0 :

                            if args.save == SAVE_ALL :
                                self.drnn_saver.save(self.sess,self.drnn_model_dir)
                            elif args.save == SAVE_DRNN:
                                self.drnn_saver.save(self.sess,self.drnn_model_dir)
                    
                    step = step+1

                time.sleep(0.001)
           

    def predict_thread(self): 
        print "prediction_thread : start"

        with self.coord.stop_on_exception():

            fRx = open("Rx.csv",'w')

            step = 0
            x_buf = []
            u_buf = []

            while not self.coord.should_stop():
                if self.stateSampled == True and self.emgSampled == True:
                    self.stateSampled = False
                    self.emgSampled = False

                    x_buf.append(self.stateData) 
                    u_buf.append(self.emgData)

                    if len(x_buf) > self.seq_length+1 :
                        x0 = np.array( x_buf[0] )
                        break

            while not self.coord.should_stop():
                if self.stateSampled == True and self.emgSampled == True:
                    self.stateSampled = False
                    self.emgSampled = False

                    x_buf.append(self.stateData)
                    u_buf.append(self.emgData)

                    x = np.array( x_buf[1:self.seq_length+1] )
                    u = np.array( u_buf[0:self.seq_length] )

                    x_buf.pop(0)
                    u_buf.pop(0) 
                    
                    feed_dict = { self.u_ph : u, 
                                  self.initial_state : np.reshape(x0, [1,self.size_x]),
                                  self.keep_prob_ph : 1.0  } 

                    x_predict_ = self.sess.run(self.x_next, feed_dict=feed_dict)
                    x_predict = np.reshape( x_predict_, [self.seq_length, self.size_x] )
                    err_x = np.sum(np.absolute(x_predict[-1,:]-x[-1,:]))

                    x0 = x_predict[0,:]
                     
                    if args.ros == 1 :
                        pub_msg = Float32MultiArray(data=x_predict[-1,:])
                    self.pubPredict.publish(pub_msg)
                    
                    print "State Err : {:.3f}".format(err_x) 
                    
                    fRx.write("{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\r\n".format(x_real[0],x_real[1],x_real[2],x_real[3],x_real[4],
                                                                                               x_predict[0],x_predict[1],x_predict[2],x_predict[3],x_predict[4]))

                time.sleep(0.001)

            fRx.close()
    
    def file_process_thread(self):
        print "file_process_thread : start"

        with self.coord.stop_on_exception():

            # Non-ROS : get data from files
            if args.ros == False:
                while not self.coord.should_stop():

                    self.emgData = self.sess.run( self.get_data_emg )[0]
                    self.stateData = self.sess.run( self.get_data_state )[0]

                    self.emgSampled = True
                    self.stateSampled = True
                
                    time.sleep(0.001)

            # Using ROS & Mode is data saving mode 
            elif args.mode == MODE_SAVE_DATA:
                
                fEmg = open(self.filenameHeader+"_emg.csv", 'w')
                fState = open(self.filenameHeader+"_state.csv", 'w')
                sample=0

                while not self.coord.should_stop():
                    if self.emgSampled == True and self.stateSampled == True :

                        for d in self.emgData[:-1]:
                            fEmg.write("{:f},".format(d)) 
                        fEmg.write("{:f}\n".format(self.emgData[-1]))

                        for d in self.stateData[:-1]:
                            fState.write("{:f},".format(d))
                        fState.write("{:f}\n".format(self.stateData[-1]))

                        sample = sample + 1
                        print "Sample {:d} is saved".format(sample)

                        self.emgSampled = False
                        self.stateSampled = False
                
                    time.sleep(0.001)
            else:
                return 0 

    def main(self):

        threads = [ threading.Thread(target=self.enque_thread) ,
                    threading.Thread(target=self.train_thread) ,
                    threading.Thread(target=self.predict_thread),
                    threading.Thread(target=self.file_process_thread) ] + self.que_thread   

        self.coord.register_thread( threads[0] )
        self.coord.register_thread( threads[1] )
        self.coord.register_thread( threads[2] )
        self.coord.register_thread( threads[3] )
       # self.coord.register_thread( threads[4] )

        # Using ROS : get data from ros messages 
        if args.ros == True :
            #Start ros
            rospy.init_node('Estimator')
            rate = rospy.Rate(500)
        
            #Start threads
            if args.mode == MODE_PREDICTION:
                threads[2].start() #Predict thread
            elif args.mode == MODE_SAVE_DATA:
                threads[3].start() #File process(writing) thread
            else : 
                threads[0].start() #Enque thread
                threads[1].start() #Train thread

            #Run
            with self.coord.stop_on_exception():
                while not rospy.is_shutdown() and not self.coord.should_stop():
                    rate.sleep()
        
        else : # non-ROS : get data from files

            #Start threads
            if args.mode == MODE_PREDICTION:
                threads[3].start() #File process(read) start
                threads[2].start() #Predict thread
            elif args.mode == MODE_SAVE_DATA:
                print "Can not save data in non-ros configuration!"
                return 0
            else :
                threads[3].start() #File process(read) start
                threads[0].start() #Enque thread
                threads[1].start() #Train thread

            #Run
            with self.coord.stop_on_exception():
                while not self.coord.should_stop():
                    time.sleep(0.002)
            
        #Stop this program
        self.sess.run( self.queClose_op )
        self.coord.request_stop()
        self.coord.join(threads) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '-mode',
            type=int,
            default=0,
            help="(INT) Mode configuration [ 0: prediction(default) , 1: train all, 2: train AE, 3: train DRNN, 4: data save ]"
            )
    
    parser.add_argument(
            '-ros',
            type=int,
            default=1,
            help="(BOOL) Use ros if it is 1(default)"
    )

    parser.add_argument(
            '-filename',
            type=str,
            default="default",
            help="(STR) File name header => Data will be saved in 'filenameHeader'_emg.csv ( Default : 'default' )" 
    )
   
    parser.add_argument(
            '-log_dir',
            type=str,
            default='/home/taeho/catkin_ws/src/bhand/src/tf_log',
            help="(STR) Logging directory ( default : tf_log )"
            )

    parser.add_argument(
            '-load',
            type=int,
            default=0,
            help="(INT) Load model( Do not Load : 0(defalut), Load all : 1, Load AE: 2, Load DRNN: 3 )"
            )

    parser.add_argument(
            '-save',
            type=int,
            default=0,
            help="(INT) Save model( Do not Save : 0(default), Save all : 1, Save AE: 2, Save DRNN: 3 )"
            )

    args, unparsed = parser.parse_known_args()

    tensor = RosTF()
    tensor.main()
