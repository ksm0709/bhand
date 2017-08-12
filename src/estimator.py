#!/usr/bin/env python  
import sys
import time
import argparse
import math

import tensorflow as tf
import numpy as np
import threading

import rospy
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Int8

MODE_PREDICTION = 0
MODE_LEARNING = 1
MODE_TEST = 2

args = None

def read_csv(filename_queue):

    reader = tf.TextLineReader()

    key, value = reader.read(filename_queue)

    record_defaults = [ [1] for _ in range(args.layer_in + args.layer_out) ]
  
    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13 = tf.decode_csv( value, record_defaults = record_defaults )

    data_in = tf.stack( [c1,c2,c3,c4,c5,c6,c7,c8] )
    data_out = tf.stack( [c9,c10,c11,c12,c13] )

    return data_in, data_out

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

def inference(signal):

    layer_h = 8

    hidden1 = nn_layer(signal, args.layer_in, layer_h, 'hidden1', act=tf.nn.relu)
    hidden2 = nn_layer(hidden1, layer_h, layer_h , 'hidden2', act=tf.nn.relu)
    
    fout = nn_layer(hidden2, layer_h, args.layer_out, 'fout', act=None)
    
#    sfm = tf.nn.softmax(fout)

    keep_prob = tf.constant(1.0)
    result = tf.nn.dropout(fout,keep_prob=keep_prob) 
    
    return result 

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

class RosTF():
    def __init__(self):
      
        ###Common Vars
        self.emgSampled = False
        self.gloveSampled = False
        
        #if not Test mode
        if not args.mode == MODE_TEST :
            
            ###ROS Init
            self.subEmg = rospy.Subscriber('actEMG',
                    Int8MultiArray,
                    self.callbackEmg, 
                    queue_size=1)

            self.subGlove = rospy.Subscriber('normGlove',
                    Int8MultiArray,
                    self.callbackGlove, 
                    queue_size=1)

            self.pubPosture = rospy.Publisher('posture',
                    Int8,
                    queue_size=10)


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
                self.signal_ph = tf.placeholder(tf.float32, shape=[args.batch_size, args.layer_in])
                self.freal_ph = tf.placeholder(tf.float32, shape=[args.batch_size, args.layer_out])
                self.signalEnque_ph =  tf.placeholder(tf.float32, shape=[args.layer_in])
                self.frealEnque_ph = tf.placeholder(tf.float32, shape=[args.layer_out])
                self.array4_ph = tf.placeholder(tf.int32, shape=[4])

                #NN
                self.fout = inference( self.signal_ph )         # for training 
                self.loss = loss_function( self.fout, self.freal_ph )
                self.train_op = training( self.loss, args.learning_rate ) 
                self.fout_softmax = tf.nn.softmax( self.fout )
                self.fout_index = tf.argmax( self.fout_softmax,1 )
                self.fout_index_mean = tf.reduce_mean( self.fout_index )
                self.freal_index = tf.argmax( self.freal_ph,1 )
                self.largest = tf.arg_max( self.array4_ph,0 ) 
                   
                #Summary
                self.summary = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

                #Init Variable
                self.init = tf.global_variables_initializer()
                self.sess.run(self.init)

                #Saver
                self.saver = tf.train.Saver()
                # Load Model ################################################# 
                self.model_dir = "tf_model/model_posture4.ckpt"
                self.saver.restore(self.sess,self.model_dir)
               ############################################################### 

                #Queue & Coordinator
                self.coord = tf.train.Coordinator()
                #self.que = tf.FIFOQueue(shapes=[[args.layer_in],[args.layer_out]],
                #                        dtypes=[tf.float32, tf.float32],
                #                        capacity=100)
                self.que = tf.RandomShuffleQueue(shapes=[[args.layer_in],[args.layer_out]],
                                                 dtypes=[tf.float32, tf.float32],
                                                 capacity=100, min_after_dequeue=20, seed=2121)


                self.queClose_op = self.que.close(cancel_pending_enqueues=True)
                self.enque_op = self.que.enqueue([self.signalEnque_ph, self.frealEnque_ph])
                self.deque_op = self.que.dequeue_many(args.batch_size) 
               
                #Dataset readers & Input pipeline
                fq_train = tf.train.string_input_producer( ["tf_data/posture4/1.csv",
                                                            "tf_data/posture4/2.csv",
                                                            "tf_data/posture4/3.csv",
                                                            "tf_data/posture4/4.csv"] )

                fq_test = tf.train.string_input_producer(["tf_data/posture4/1_handopen_3.csv",
                                                          "tf_data/posture4/2_indexpoint_3.csv",
                                                          "tf_data/posture4/3_fistgrip_3.csv",
                                                          "tf_data/posture4/4_tripodgrip_3.csv"])

                self.x_test, self.y_test = read_csv(fq_test)
                self.x_train, self.y_train = read_csv(fq_train)

                self.queThread = tf.train.start_queue_runners(sess=self.sess, coord=self.coord) 
                
                print "-------------------------------------------------"

    def callbackEmg(self, msg): 
        self.emgData = msg.data
        self.emgSampled = True

    def callbackGlove(self, msg):
        self.gloveData = msg.data
        self.gloveSampled = True
    
    def enque_thread(self):
        print "enque_thread : start"

        with self.coord.stop_on_exception():

            if args.mode == MODE_LEARNING or args.mode == MODE_PREDICTION:
             
                while not self.coord.should_stop():
                    if self.emgSampled == True :# and self.gloveSampled == True :
                        
                        self.gloveData = (0,0,0,0,0)
                        
                        x = np.array(self.emgData) 
                        y = np.array(self.gloveData)

                        self.sess.run(self.enque_op, feed_dict={ self.signalEnque_ph : x,
                                                                 self.frealEnque_ph : y } )
                        self.emgSampled = False
                        self.gloveSampled = False

                    time.sleep(0.001)

            elif args.mode == MODE_TEST:
                while not self.coord.should_stop():
                    x, y = self.sess.run([self.x_train, self.y_train]) # read one dataset 

                    self.sess.run(self.enque_op, feed_dict={ self.signalEnque_ph : x,
                                                         self.frealEnque_ph : y } )

                    time.sleep(0.001)

    def train_thread(self):
        print "train_thread : start"

        with self.coord.stop_on_exception():

            step = 0
            while not self.coord.should_stop():
                x, y = self.sess.run( self.deque_op )
               
                feed_dict = { self.signal_ph : x, self.freal_ph : y } 

                _,self.result,freal_index, loss = self.sess.run([self.train_op, self.fout_index, self.freal_index,  self.loss], feed_dict=feed_dict)

                step = step+1

                if step % 10 == 0 :

                    print "step : {0}   loss : {1}   freal,fout : \n{2}\n{3}".format(step, loss, freal_index,self.result)

                    summary_str = self.sess.run( self.summary, feed_dict = feed_dict )
                    self.summary_writer.add_summary( summary_str, step )
                    self.summary_writer.flush()
                    
                    if step%100 == 0:
                        self.saver.save(self.sess, self.model_dir) 

                time.sleep(0.001)
           

    def predict_thread(self): 
        print "prediction_thread : start"

        with self.coord.stop_on_exception():

            step = 0
            cnt = 0 
            buf = [] 
            while not self.coord.should_stop():
                #x, _ = self.sess.run( self.deque_op )     

                if cnt < args.batch_size :
                    
                    if self.emgSampled == True:
                        self.emgSampled = False
                        
                        buf.append(list(self.emgData))
                   
                        cnt = cnt+1
                
                else:
                    x = np.array( buf )

                    feed_dict = { self.signal_ph : x } 

                    self.result = self.sess.run(self.fout_index, feed_dict=feed_dict)

                    step = step + 1

                    if step%10 == 0:

                        count=[0,0,0,0]
                        for r in self.result:
                           count[r] = count[r]+1
                        
                        idx = self.sess.run( self.largest, feed_dict = { self.array4_ph : count })

                        self.pubPosture.publish( idx )
                        
                        print "Predicted posture : {0}".format(idx)

                    cnt = 0
                    buf = []

                time.sleep(0.001)


    def main(self):

        threads = [ threading.Thread(target=self.enque_thread) ,
                    threading.Thread(target=self.train_thread) ,
                    threading.Thread(target=self.predict_thread) ,
                    self.queThread ]    

        self.coord.register_thread( threads[0] )
        self.coord.register_thread( threads[1] )
        self.coord.register_thread( threads[2] )

        
        # Train thread or Predict thread start
        if args.mode == MODE_TEST:
            threads[0].start()
            threads[1].start()

            with self.coord.stop_on_exception():
                try:
                    while not self.coord.should_stop():
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    self.coord.request_stop();

        elif args.mode == MODE_LEARNING:
            threads[0].start()
            threads[1].start()
        
            rospy.init_node('Estimator')
            rate = rospy.Rate(100)            

            with self.coord.stop_on_exception():
                while not rospy.is_shutdown() and not self.coord.should_stop():
                    rate.sleep() 
        
        else : 
            threads[2].start()
           
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
            help="Mode configuration [ 0: prediction only(default) , 1: prediction with learning , 2: test mode( ROS offline, use CSV files ) ]"
            )
   
    parser.add_argument(
            '-layer_in',
            type=int,
            default=8,
            help="Input layer size"
            )

    parser.add_argument(
            '-layer_out',
            type=int,
            default=5,
            help="Output layer size"
            )

    parser.add_argument(
            '-layer_h1',
            type=int,
            default=8,
            help="Hidden layer 1 size"
            )

    parser.add_argument(
            '-layer_h2',
            type=int,
            default=8,
            help="Hidden layer 2 size"
            )

    parser.add_argument(
            '-learning_rate',
            type=float,
            default=0.01,
            help="Learning rate ( default : 0.01 )"
            )

    parser.add_argument(
            '-batch_size',
            type=int,
            default=20,
            help="Batch size ( default : 20 )"
            )

    parser.add_argument(
            '-log_dir',
            type=str,
            default='tf_log',
            help="Logging directory ( default : tf_log )"
            )
    

    args, unparsed = parser.parse_known_args()

    tensor = RosTF()
    tensor.main()
