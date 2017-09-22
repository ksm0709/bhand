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
        self.size_zx = 3
        self.size_zu = 5
        self.size_x = 5
        self.size_u = 8
        self.layer_in = self.size_zx + self.size_zu
        self.layer_out = self.size_zx
        self.batch_size = 50
        self.learning_rate = 0.01
        self.seq_length = 100
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
                self.zu_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_zu]) # Embeded U
                self.zx_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_zx]) # Embeded X
                self.u_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_u])   # U (emg)
                self.x_ph = tf.placeholder(tf.float32, shape=[None, None, self.size_x])   # X (state)
                self.uEnque_ph = tf.placeholder(tf.float32, shape=[None, self.size_u]) 
                self.xEnque_ph = tf.placeholder(tf.float32, shape=[None, self.size_x])
                self.x0Enque_ph = tf.placeholder(tf.float32, shape=[self.size_x])
                self.initial_state = tf.placeholder(tf.float32, shape=[None,self.size_zx])
                self.drnn_dropout = tf.placeholder(tf.float32)

                #Autoencoder for EMG
                with tf.variable_scope("AE_u"):
                    self.zu, self.ru = Autoencoder(self.u_ph, range(self.size_u,self.size_zu-1,-1),act = tf.nn.elu, 
                                                                                                   keep_prob=1.0,
                                                                                                   l2_reg=0.01)

                    zu_mean = tf.reduce_mean(self.zu, axis=0)
                    sparsity_loss = self.sparsity_weight * tf.reduce_sum(kl_divergence(self.sparsity_target, zu_mean))
                    self.train_zu, self.loss_zu = set_optimizer(self.u_ph, self.ru, self.learning_rate, ADAM_OPTIMIZER, scope="AE_u", add_loss=sparsity_loss)
                    tf.summary.scalar( 'ae_u_loss' , self.loss_zu[0] )
                    tf.summary.scalar( 'ae_u_loss_mse', self.loss_zu[1])
                    tf.summary.scalar( 'ae_u_loss_reg', self.loss_zu[2])
                    tf.summary.scalar( 'ae_u_loss_sparsity', self.loss_zu[3])

                    #tf.summary.image('u_in', tf.reshape(self.u_ph,[1,1,self.size_u,1]),1)
                    # tf.summary.image('ae_u_zu',tf.reshape(self.zu,[1,1,self.size_zu,1]),1)
                    # tf.summary.image('u_out', tf.reshape(self.ru, [1,1,self.size_u,1]),1)

                #Autoencoder for Glove
                with tf.variable_scope("AE_x"):
                    self.zx, self.rx = Autoencoder(self.x_ph, range(self.size_x,self.size_zx-1,-1),act = tf.nn.elu, 
                                                                                                   keep_prob=1.0,
                                                                                                   l2_reg=0.01)
                    zx_mean = tf.reduce_mean(self.zx, axis=0)
                    sparsity_loss = self.sparsity_weight * tf.reduce_sum(kl_divergence(self.sparsity_target, zx_mean))
                    self.train_zx, self.loss_zx = set_optimizer(self.x_ph, self.rx, self.learning_rate, ADAM_OPTIMIZER, scope="AE_x", add_loss=sparsity_loss)
                    tf.summary.scalar( 'ae_x_loss' , self.loss_zx[0] )
                    tf.summary.scalar( 'ae_x_loss_mse', self.loss_zx[1])
                    tf.summary.scalar( 'ae_x_loss_reg', self.loss_zx[2])
                    tf.summary.scalar( 'ae_x_loss_sparsity', self.loss_zx[3])

                    #tf.summary.image('x_in',tf.reshape(self.x_ph,[1,1,self.size_x,1]),1)
                    # tf.summary.image('ae_x_zx',tf.reshape(self.zx,[1,1,self.size_zx,1]),1)
                    # tf.summary.image('x_out',tf.reshape(self.rx, [1,1,self.size_x,1]),1)

                #DRNNNetwork
                with tf.variable_scope("DRNN"):
                    self.cell = DRNNCell(num_output=self.layer_out, num_units=[30,40,30], activation=tf.nn.elu, output_activation=tf.nn.sigmoid, keep_prob=self.drnn_dropout) 
                    self.lstm_cell = tf.contrib.rnn.GRUCell(num_units=self.layer_out)
                    self.lstm_init = tf.contrib.rnn.LSTMStateTuple(self.initial_state,self.initial_state)
                    self.zx_next, _states = tf.nn.dynamic_rnn( self.lstm_cell, self.zu_ph, initial_state= self.initial_state, dtype=tf.float32 )

                    self.train_drnn, self.loss_drnn = set_optimizer(self.zx_ph, self.zx_next, self.learning_rate, scope="DRNN")
                    tf.summary.scalar( 'drnn_loss' , self.loss_drnn[0] )
                    tf.summary.scalar( 'drnn_loss_mse', self.loss_drnn[1])
                    tf.summary.scalar( 'drnn_loss_reg', self.loss_drnn[2])

                # Dict of train functions 
                self.train_func = {
                    MODE_TRAIN_AE : self.func_train_ae,
                    MODE_TRAIN_DRNN : self.func_train_drnn,
                    MODE_TRAIN_ALL : self.func_train_all,
                    }

                #Init Variable
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)
                
                # Saver / Load Model ######################################### 
                self.ae_x_model_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_model/model_AEx.ckpt"
                self.ae_u_model_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_model/model_AEu.ckpt"
                self.drnn_model_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_model/model_DRNN.ckpt"
                
                self.ae_x_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AE_x"))
                self.ae_u_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AE_u"))
                self.drnn_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DRNN"))

                if args.load == LOAD_ALL :
                    self.ae_x_saver.restore(self.sess,self.ae_x_model_dir)
                    self.ae_u_saver.restore(self.sess,self.ae_u_model_dir)
                    self.drnn_saver.restore(self.sess,self.drnn_model_dir)
                elif args.load == LOAD_AE:
                    self.ae_x_saver.restore(self.sess,self.ae_x_model_dir)
                    self.ae_u_saver.restore(self.sess,self.ae_u_model_dir)
                elif args.load == LOAD_DRNN:
                    self.drnn_saver.restore(self.sess,self.drnn_model_dir)

               ############################################################### 

                #Queue & Coordinator
                self.coord = tf.train.Coordinator()
                #self.que = tf.FIFOQueue(shapes=[[self.seq_length,self.size_u],[self.seq_length,self.size_x],[self.size_x]], # u, x, x0
                #                       dtypes=[tf.float32, tf.float32, tf.float32],
                #                       capacity=1000)
                self.que = tf.RandomShuffleQueue(shapes=[[self.seq_length,self.size_u],[self.seq_length,self.size_x],[self.size_x]],
                                                 dtypes=[tf.float32, tf.float32, tf.float32],
                                                 capacity=3000, min_after_dequeue=2000, seed=2121)


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
    
    

    def func_train_all(self,step,feed_dict):
        _,_,_,loss_zx,loss_zu,loss_drnn = self.sess.run([self.train_zx, 
                                                         self.train_zu, 
                                                         self.train_drnn, 
                                                         self.loss_zx, 
                                                         self.loss_zu, 
                                                         self.loss_drnn], 
                                                         feed_dict=feed_dict)
                                                            
        if step % 10 == 0:
           print "step : {:d}   loss_AE_x : {:5.5f}    loss_AE_u : {:5.5f}    loss_drnn : {:5.5f}".format(step,loss_zx[0], loss_zu[0], loss_drnn[0])

        return 0

    def func_train_ae(self,step,feed_dict):
        _,_,loss_zx,loss_zu = self.sess.run([self.train_zx, 
                                             self.train_zu, 
                                             self.loss_zx, 
                                             self.loss_zu], feed_dict=feed_dict)

        if step % 10 == 0 :
            print "step : {:d}   loss_AE_x : {:5.5f}    loss_AE_u : {:5.5f}".format(step,loss_zx[0], loss_zu[0])

        return 0
    def func_train_drnn(self,step,feed_dict):

        _,loss_drnn = self.sess.run([self.train_drnn, self.loss_drnn], feed_dict=feed_dict)

        if step % 10 == 0 :

            print "step : {:d}   loss_drnn : {:5.5f}".format(step, loss_drnn[0])

           

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

                    # x0 reshape cosidering sequence length dim
                    x0 = np.reshape(x0, [self.batch_size,1,self.size_x])

                    # Encoding
                    zu = self.sess.run( self.zu, feed_dict={ self.u_ph : u })
                    zx = self.sess.run( self.zx , feed_dict={ self.x_ph : x })
                    zx0 = self.sess.run( self.zx , feed_dict={ self.x_ph : x0 })

                    # zx0 reshape removing sequence length dim
                    zx0 = np.reshape(zx0, [self.batch_size,self.size_zx])
                  
                    feed_dict = { self.u_ph : u,
                                  self.x_ph : x,
                                  self.zu_ph : zu, 
                                  self.zx_ph : zx, 
                                  self.initial_state : zx0,
                                  self.drnn_dropout : 0.8
                                  } 
                    
                    # Training
                    self.train_func[args.mode](step,feed_dict)

                    # Summary & Save
                    if step%10 == 0:
                        summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
                        self.summary_writer.add_summary( summary_str, step )
                        self.summary_writer.flush()
            
                        if step%1000 == 0 :

                            if args.save == SAVE_ALL :
                                self.ae_x_saver.save(self.sess,self.ae_x_model_dir)
                                self.ae_u_saver.save(self.sess,self.ae_u_model_dir)
                                self.drnn_saver.save(self.sess,self.drnn_model_dir)
                            elif args.save == SAVE_AE:
                                self.ae_x_saver.save(self.sess,self.ae_x_model_dir)
                                self.ae_u_saver.save(self.sess,self.ae_u_model_dir)
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

                    # Buffering Data
                    x_buf.append(self.stateData)
                    u_buf.append(self.emgData)

                    x = np.array( x_buf[1:self.seq_length+1] )
                    u = np.array( u_buf[0:self.seq_length] )

                    x_buf.pop(0)
                    u_buf.pop(0) 

                    # Get initial state , input
                    zu = self.zu.eval(session=self.sess, feed_dict={ self.u_ph : np.reshape(u,[1,self.seq_length,self.size_u])})
                    zx0 = self.zx.eval(session=self.sess, feed_dict={ self.x_ph : np.reshape(x0,[1,1,self.size_x])})
                    
                    # Predict & print
                    feed_dict = { self.zu_ph : zu, 
                                  self.initial_state : np.reshape(zx0,[1,self.size_zx]),
                                  self.drnn_dropout : 1.0  } 

                    zx_predict_ = self.sess.run(self.zx_next, feed_dict=feed_dict)
                    x_predict_ = self.rx.eval(session=self.sess,feed_dict={self.zx : zx_predict_})
                    x_predict = np.reshape( x_predict_, [self.seq_length, self.size_x] )
                    err_x = np.sum(np.absolute(x_predict[-1,:]-x[-1,:]))

                    x0 = x_predict[0,:]
                     
                    if args.ros == 1 :
                        pub_msg = Float32MultiArray(data=x_predict[-1,:])
                        self.pubPredict.publish(pub_msg)
                    
                    print "State Err : {:.3f}".format(err_x) 
                    
                    fRx.write("{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\r\n".format(x[-1,0],x[-1,1],x[-1,2],x[-1,3],x[-1,4],
                                                                                               x_predict[-1,0],x_predict[-1,1],x_predict[-1,2],x_predict[-1,3],x_predict[-1,4]))

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
