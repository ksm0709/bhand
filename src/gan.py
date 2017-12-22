#!/usr/bin/env python  
import sys
import time
import argparse
import math

import tensorflow as tf
import numpy as np
import threading
import Queue
from os import listdir

import rospy
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Int8
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import l2_regularizer

np.set_printoptions(threshold=np.inf)

MODE_PREDICTION = 0
MODE_TRAINING = 1
SAVE_ALL = 1
LOAD_ALL = 1

class RosTF():
    def __init__(self):
      
        ###Common Vars
        self.train_done = False
        self.predict_done = False
        self.emgSampled = False
        self.filenameHeader = args.filename
        self.use_z = False
        self.use_mse = False
        self.use_dz = True
        self.buf=[]
        self.buf_flag = False

        ###Network Vars(Hyper Parameters)
        self.n_z = 10
        self.n_emg = 8
        self.n_domain = 16
        self.n_class = 10
        
        if self.use_z == True :
            self.n_input = self.n_emg + self.n_domain + self.n_z
            self.n_features = self.n_emg + self.n_domain
        else :
            self.n_input = self.n_emg + self.n_domain
            self.n_features = self.n_emg + self.n_domain

        self.n_rep = 4
        self.n_data = self.n_features + self.n_class

        self.n_G_input = self.n_input
        self.n_G_output = self.n_features

        self.n_D_input = self.n_features
        self.n_D_output = 1 + self.n_class  

        self.n_ref = 0
        self.n_buf = 3
        self.n_batch = 250
        self.learning_rate = 0.005
        self.lamb_c = 1.0
        self.lamb_d = 2.0
        self.lamb_f = 2.0
        self.sigma_dz = 0.2
                
        self.current_d = np.array( [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] )

        degree = input('Placement degree? ')
        d0 = (degree*(16.0/360.0))%16.0
        db = d0 - int(d0)
        da = 1.0 - db
        self.current_d[int(d0)] = da
        self.current_d[int(d0)+1] = db 
        self.current_d = np.correlate(self.current_d,[0.0,1.0,0.0],"same")
        self.current_d_batch = np.array( [self.current_d for _ in range(self.n_batch)])
        print self.current_d

        self.ref_target_X = []
        self.ref_target_Y = []

        ###ROS Init
        if args.ros == True:
            rospy.init_node('GAN')
            self.subEmg = rospy.Subscriber('/actEMG',
                    Float32MultiArray,
                    self.callbackEmg,
                    queue_size=1)
            
            self.pubPredict = rospy.Publisher('/posture',
                    Int8,
                    queue_size=1)

            self.loop_rate = rospy.Rate(1000)

            #self.get_ref()
            
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

                with tf.name_scope("Placeholder_and_Variables"): 
                    # placeholder 
                    self.X_normal = tf.placeholder(shape=[None, self.n_features], dtype=tf.float32)
                    self.Y_normal = tf.placeholder(shape=[None, self.n_class], dtype=tf.float32)
                    self.X_source = tf.placeholder(shape=[None, self.n_features], dtype=tf.float32)
                    self.Y_source = tf.placeholder(shape=[None, self.n_class], dtype=tf.float32)
                    self.z_prior = tf.placeholder(shape=[None, self.n_z], dtype=tf.float32)
                    self.dz = tf.placeholder(shape=[None,self.n_domain], dtype=tf.float32)
                    self.d_target = tf.placeholder(shape=[None, self.n_domain], dtype=tf.float32)
                    self.d_normal = tf.constant([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in range(self.n_batch)], dtype=tf.float32)
                    self.keep_prob = tf.placeholder(tf.float32)
                    
                    with tf.variable_scope("Estimated"):
                        self.a_target = tf.placeholder(shape=[None, self.n_emg], dtype=tf.float32) 
                        self.d_target_e1 = tf.nn.softmax( tf.Variable(tf.zeros([self.n_domain]),trainable=True) )
                        self.d_target_e = tf.tile( self.d_target_e1, [self.n_batch] )
                        self.d_target_e = tf.reshape( self.d_target_e, [self.n_batch, self.n_domain])
                        self.X_target_e = tf.concat([self.a_target, self.d_target_e],1) 

                    self.X_target = tf.placeholder(shape=[None, self.n_features], dtype=tf.float32) 
                    self.Y_target = tf.placeholder(shape=[None, self.n_class], dtype=tf.float32)

                    self.phase = tf.placeholder(dtype=tf.bool, name='phase')

                    self.a_enque_ph = tf.placeholder(tf.float32, shape=[self.n_emg])
                    self.d_enque_ph = tf.placeholder(tf.float32, shape=[self.n_domain])
                    self.y_enque_ph = tf.placeholder(tf.float32, shape=[self.n_class])
                    self.bn_phase = tf.placeholder(tf.bool)

                    
                # Input Pipline 
                with tf.name_scope("Input_Pipeline"):
                    normal_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_data/thesis/normal/s0"
                    source_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_data/thesis/source/s0"
                    target_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_data/thesis/target"
                    list_filename_normal=[normal_dir + '/' + f for f in listdir(normal_dir)]
                    list_filename_source=[source_dir + '/' + f for f in listdir(source_dir)]
                    list_filename_target=[target_dir + '/' + f for f in listdir(target_dir)]

                    self.sample_batch_normal = self.input_pipline(list_filename_normal, data_size=self.n_data, batch_size=self.n_batch)
                    self.sample_batch_source = self.input_pipline(list_filename_source, data_size=self.n_data, batch_size=self.n_batch)
                    self.sample_batch_target = self.input_pipline(list_filename_target, data_size=self.n_data, batch_size=self.n_batch, is_target=True)

                # Define GAN network
                with tf.name_scope("GAN"):
                        # From normal
                    D_normal_l, D_normal_c, D_normal_d, F_normal = self.model_D(self.X_normal, phase=self.phase, scope="Discriminator")
                    
                    if self.use_z == False :   
                        # From source
                        G_source, G_source_dl = self.model_G( self.X_source, phase=self.phase, scope="Generator")
                        D_source_l, D_source_c, D_source_d, F_source = self.model_D(G_source, reuse=True, phase=self.phase, scope="Discriminator")

                        # From target
                        G_target, G_target_dl = self.model_G( self.X_target, reuse=True, phase=self.phase, scope="Generator")
                        D_target_l, D_target_c, D_target_d, F_target = self.model_D(G_target,reuse=True, phase=self.phase, scope="Discriminator")
                        

                        # From target_e : estimated d_target
                        Ge_target, Ge_target_dl = self.model_G( self.X_target_e, reuse=True, phase=self.phase, scope="Generator")
                        De_target_l, De_target_c, De_target_d, Fe_target = self.model_D(Ge_target,reuse=True, phase=self.phase, scope="Discriminator")
                    else :
                        ############## WITH LATENT SPACE Z ##############################
                        # From source
                        G_source, G_source_dl = self.model_G( tf.concat( [self.X_source, self.z_prior],1 ), phase=self.phase, scope="Generator")
                        D_source_l, D_source_c, D_source_d, F_source = self.model_D(G_source, reuse=True, phase=self.phase, scope="Discriminator")

                        # From target
                        G_target, G_target_dl = self.model_G( tf.concat( [self.X_target, self.z_prior],1 ), reuse=True, phase=self.phase, scope="Generator")
                        D_target_l, D_target_c, D_target_d, F_target = self.model_D(G_target,reuse=True, phase=self.phase, scope="Discriminator")

                        # From target_e : estimated d_target
                        Ge_target, Ge_target_dl = self.model_G( tf.concat( [self.X_target_e, self.z_prior],1 ), reuse=True, phase=self.phase, scope="Generator")
                        De_target_l, De_target_c, De_target_d, Fe_target = self.model_D(Ge_target,reuse=True, phase=self.phase, scope="Discriminator")
                        ##################################################################

                    # Get Network Vars
                    all_vars= tf.global_variables()
                    d_var = self.get_var(all_vars, "Estimated")
                    G_var = self.get_var(all_vars, "Generator")
                    D_var = self.get_var(all_vars, "Discriminator") 

                with tf.name_scope("gan_realtime"):
                    G_realtime, G_realtime_dl = self.model_G( self.X_target, phase=self.phase, scope="Generator_r")
                    D_realtime_l, D_realtime_c, D_target_d, F_target = self.model_D(G_realtime, phase=self.phase, scope="Discriminator_r")
                    Dd_target_l, Dd_target_c, Dd_target_d, Fd_target = self.model_D(self.X_target, reuse=True, phase=self.phase, scope="Discriminator_r")

                    # Get Network Vars
                    all_vars= tf.global_variables()
                    Gr_var = self.get_var(all_vars, "Generator_r")
                    Dr_var = self.get_var(all_vars, "Discriminator_r") 

                    self.update_network_params = [Gr_var[i].assign(G_var[i]) for i in range(len(Gr_var))] + [Dr_var[j].assign(D_var[j]) for j in range(len(Dr_var)) ]            

                with tf.name_scope("ops"): 
                    # Loss & Train op

                    if self.use_mse == True:
                        # Without Target
                        #self.loss_D = tf.reduce_mean(tf.square(1-D_normal_d)) + tf.reduce_mean(tf.square(D_source_d)) + self.lamb*tf.reduce_mean( self.cross_entropy(self.Y_normal, D_normal_l) )
                        #self.loss_G = tf.reduce_mean(tf.square(1-D_source_d)) + self.lamb*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l)) + tf.reduce_mean(tf.square(F_normal-F_source)) + tf.reduce_mean(self.cross_entropy(self.d_normal, G_source_dl)) 

                        # With Target
                        self.loss_D = tf.reduce_mean(tf.square(1-D_normal_d)) + tf.reduce_mean(tf.square(D_target_d)) + tf.reduce_mean(tf.square(D_source_d)) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_normal, D_normal_l) ) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l) ) 
                        self.loss_G = tf.reduce_mean(tf.square(1-D_target_d)) + tf.reduce_mean(tf.square(1-D_source_d)) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l)) + self.lamb_f*tf.reduce_mean(tf.square(F_normal-F_target)) + self.lamb_f*tf.reduce_mean(tf.square(F_source-F_target))+ self.lamb_d*tf.reduce_mean(self.cross_entropy(self.d_normal, G_source_dl)) + self.lamb_d*tf.reduce_mean(self.cross_entropy(self.d_normal, G_target_dl)) 

                    else :
                        # Without Target
                        # self.loss_D = tf.reduce_mean(-tf.log(D_normal_d)) + tf.reduce_mean(-tf.log(1-D_source_d)) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_normal, D_normal_l) ) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l) ) 
                        # self.loss_G = tf.reduce_mean(-tf.log(D_source_d)) +  self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l)) + self.lamb_d*tf.reduce_mean(self.cross_entropy(self.d_normal, G_source_dl))

                        # With Target
                        self.loss_D = tf.reduce_mean(-tf.log(D_normal_d)) + tf.reduce_mean(-tf.log(1-D_target_d)) + tf.reduce_mean(-tf.log(1-D_source_d)) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_normal, D_normal_l) ) + self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l) ) 
                        self.loss_G = tf.reduce_mean(-tf.log(D_target_d)) + tf.reduce_mean(-tf.log(D_source_d)) +  self.lamb_c*tf.reduce_mean( self.cross_entropy(self.Y_source, D_source_l)) + self.lamb_f*tf.reduce_mean(tf.square(F_normal-F_target)) + self.lamb_f*tf.reduce_mean(tf.square(F_source-F_target)) + self.lamb_d*tf.reduce_mean(self.cross_entropy(self.d_normal, G_source_dl)) + self.lamb_d*tf.reduce_mean(self.cross_entropy(self.d_normal, G_target_dl)) 

                    self.loss_d = tf.reduce_mean(-tf.log(De_target_d)) + tf.reduce_mean(-self.cross_entropy(De_target_c,De_target_c)) 

                    self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate) 
                    self.optimizer_SGD = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.1)
                    self.train_D = self.optimizer_SGD.minimize(self.loss_D,var_list=D_var)
                    self.train_G = self.optimizer_Adam.minimize(self.loss_G,var_list=G_var)
                    self.train_d = self.optimizer_Adam.minimize(self.loss_d,var_list=d_var)
                    self.acc_n = tf.reduce_mean( self.accuracy_measure(D_normal_c,self.Y_normal) )
                    self.acc_s = tf.reduce_mean( self.accuracy_measure(D_source_c,self.Y_source) )
                    self.acc_t = tf.reduce_mean( self.accuracy_measure(D_target_c,self.Y_target) )
                    self.extract_posture = tf.reduce_sum(D_normal_c,axis=0) + 1
                    self.data_transferred = G_realtime

                    #Init Variable
                    self.init = tf.global_variables_initializer()
                    self.sess.run(self.init)
                    
                    
                    #Queue & Coordinator
                    self.coord = tf.train.Coordinator()
                    self.que = tf.RandomShuffleQueue(shapes=[[self.n_features],[self.n_class]],
                                            dtypes=[tf.float32,tf.float32],
                                            capacity=10000, min_after_dequeue=5000, seed=2121)


                    self.queClose_op = self.que.close(cancel_pending_enqueues=True)
                    self.enque_op = self.que.enqueue([ tf.concat([self.a_enque_ph, self.d_enque_ph],0), self.y_enque_ph ])
                    self.deque_op = self.que.dequeue_many(self.n_batch) 
                    self.que_size = self.que.size()
                    self.que_thread = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
                
                    # Onehot decode for filesave
                    self.decode_normal = self.decode_onehot(self.X_normal, self.Y_normal)
                    self.decode_source = self.decode_onehot(self.X_source, self.Y_source)
                    self.decode_source_t = self.decode_onehot(G_source, D_source_c)
                    self.decode_target = self.decode_onehot(G_target, D_target_c)

                with tf.name_scope("Saver_and_Summary"):
                    # Saver / Load Model ######################################### 
                    self.gan_model_dir = "/home/taeho/catkin_ws/src/bhand/src/tf_model/model_gan.ckpt"
                    
                    self.gan_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

                    if args.load :
                        self.gan_saver.restore(self.sess,self.gan_model_dir)
                    if args.save == SAVE_ALL :
                        self.gan_saver.save(self.sess,self.gan_model_dir)
                    ############################################################### 
 
                      

                    #Summary
                    self.lossD_sum = tf.summary.scalar("Discriminator Loss", self.loss_D)
                    self.lossG_sum = tf.summary.scalar("Generator Loss", self.loss_G)
                    self.lossd_sum = tf.summary.scalar("Estimator Loss", self.loss_d)
                    self.acc_n_sum = tf.summary.scalar("Accuracy[Normal]", self.acc_n)
                    self.acc_s_sum = tf.summary.scalar("Accuracy[Source]", self.acc_s)
                    self.acc_t_sum = tf.summary.scalar("Accuracy[Target]", self.acc_t)

                    self.list_sum_train = [self.lossD_sum, self.lossG_sum, self.acc_n_sum, self.acc_s_sum, self.acc_t_sum]
                    self.list_sum_predict = [self.lossD_sum, self.lossG_sum]
                    self.list_sum_estimate= [self.lossd_sum]

                    self.summary_train = tf.summary.merge(self.list_sum_train)
                    self.summary_predict = tf.summary.merge(self.list_sum_predict)
                    self.summary_estimate = tf.summary.merge(self.list_sum_estimate)

                    self.summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

                print "-------------------------------------------------"
    def lrelu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def get_ref(self):
        
        n_sample = 0
        cnt = 0

        print "pinch"
        while cnt < 1250:
           if self.emgSampled == True :
               if cnt >= 500 and cnt < 1000:
                   self.ref_target_X.append(self.emgData)  
                   self.ref_target_Y.append([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.])
                   n_sample += 1
                   print n_sample
               self.emgSampled = False
               cnt += 1

        cnt = 0
        print "fist"
        while cnt < 1250:
           if self.emgSampled == True :
               if cnt >= 500 and cnt < 1000:
                   self.ref_target_X.append(self.emgData)  
                   self.ref_target_Y.append([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.])
                   n_sample += 1
                   print n_sample
               self.emgSampled = False
               cnt += 1

        cnt = 0
        print "hook"
        while cnt < 1250:
           if self.emgSampled == True :
               if cnt >= 500 and cnt < 1000:
                   self.ref_target_X.append(self.emgData)  
                   self.ref_target_Y.append([0.,0.,0.,1.,0.,0.,0.,0.,0.,0.])
                   n_sample += 1
                   print n_sample
               self.emgSampled = False
               cnt += 1
        cnt = 0
        print "pointing"
        while cnt < 1250:
           if self.emgSampled == True :
               if cnt >= 500 and cnt < 1000:
                   self.ref_target_X.append(self.emgData)  
                   self.ref_target_Y.append([0.,0.,0.,0.,1.,0.,0.,0.,0.,0.])
                   n_sample += 1
                   print n_sample
               self.emgSampled = False
               cnt += 1

        self.n_ref = n_sample
    
    def input_pipline(self,list_filename, data_size, batch_size, is_target=False):
        #File input pipeline
        filename_queue = tf.train.string_input_producer(list_filename,shuffle=True, name='filename_queue')
    
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        csv_data = tf.decode_csv(value, record_defaults=[[1.] for _ in range(data_size)])

        get_data = tf.train.shuffle_batch([csv_data],capacity=1000,min_after_dequeue=800,batch_size=batch_size, num_threads=3)
        
        Xa,Xd,Y = tf.split(get_data, [self.n_emg,self.n_domain,self.n_class],1)

        if is_target == True:
            Xd = self.d_target

        if self.use_dz == True:

            c_in = tf.reshape(Xd, [self.n_batch,self.n_domain,1])
            
            
            Xd = Xd + self.dz       

        X = tf.concat([Xa,Xd],1)
        
        return [X,Y]

    def layer(self,tensor_in, n_output, activation_fn=tf.nn.relu, use_bn=False, phase=True, reuse=False, scope=None):
        if use_bn == True :
            out = fully_connected(tensor_in, n_output, activation_fn=activation_fn, normalizer_fn=batch_norm, reuse=reuse, scope=scope)
        else:
            out = fully_connected(tensor_in, n_output, activation_fn=activation_fn, reuse=reuse, scope=scope)

        return out

    def model_G(self,tensor_in, reuse=None,phase=True, scope=None):
        n_layer = 6
        n_layer_h =int( n_layer/2 )
        n_neuron = int((self.n_G_input - self.n_G_output) / n_layer)
        h_cnt = 1
        
        G_h = tensor_in
        for i in range(1,n_layer_h+1):
            G_h =  self.layer(G_h, self.n_G_input + i*2, activation_fn=tf.nn.elu, use_bn=True,phase=phase, reuse=reuse, scope=scope+"/h"+str(h_cnt))
            h_cnt += 1
        for i in range(n_layer_h,0,-1):
            G_h =  self.layer(G_h, self.n_G_input + i*2-1 , activation_fn=tf.nn.elu, use_bn=True,phase=phase, reuse=reuse, scope=scope+"/h"+str(h_cnt))
            h_cnt += 1

        G_l = tf.nn.dropout(self.layer(G_h, self.n_G_output, activation_fn=None, use_bn=False,phase=phase, reuse=reuse, scope=scope+"/o"), keep_prob=0.5)
        
        Ga_l, Gd_l = tf.split( G_l, [self.n_emg, self.n_domain], 1)

        Ga = tf.nn.sigmoid( Ga_l )
        Gd = tf.nn.softmax( Gd_l )

        G_o = tf.concat([Ga,Gd],1)
        
        return G_o, Gd_l

    def model_D(self,tensor_in, reuse=None,phase=True, scope=None):
        n_layer = 6
        n_neuron = int((self.n_D_input - self.n_D_output) / n_layer)

        D_h = tensor_in
        Fbuf = []
        for i in range(1,n_layer+1):
            D_h =  self.layer(D_h, self.n_D_input - n_neuron*i, activation_fn=tf.nn.elu, use_bn=True,phase=phase, reuse=reuse, scope=scope+"/h"+str(i))
            Fbuf.append( D_h )
        D_l = tf.nn.dropout( self.layer(D_h, self.n_D_output, activation_fn=None, use_bn=False,phase=phase, reuse=reuse, scope=scope+"/o"), keep_prob=self.keep_prob )

        Dc_l, Dd_l = tf.split( D_l , [self.n_class,1], 1)
        
        Dc = tf.nn.softmax( Dc_l )
        
        if self.use_mse == True:
            Dd = Dd_l
        else :
            Dd = tf.nn.sigmoid( Dd_l )
        
        D_f = tf.concat(Fbuf, axis=1)
        F = tf.reduce_mean( D_f, axis=0)
        
        return Dc_l, Dc, Dd, F

    def get_var(self,all_vars, name):
        result = []
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                result.append( all_vars[i] )
        
        if not result:
            return None
        return result

    def decode_onehot(self, features, labels):
        
        signal,domains = tf.split(features, [self.n_emg,self.n_domain], axis=1)

        idx_domains = 1 + tf.cast(tf.reshape(tf.argmax(domains, axis=1),[self.n_batch,1]), dtype=tf.float32)
        idx_labels = 1 + tf.cast(tf.reshape(tf.argmax(labels, axis=1),[self.n_batch,1]), dtype=tf.float32)

        result = tf.concat([idx_labels,signal,100.0*idx_domains],axis=1)

        return result

    def normalize(self,t):
        t_sum = tf.reduce_sum(t,axis=1)
        t_tile = tf.tile(t_sum, [tf.shape(t)[1]] )
        t_res = tf.transpose( tf.reshape( t_tile, tf.shape(tf.transpose(t)) ) )
    
        return t / t_res

    def cross_entropy(self,p, q):
        return tf.nn.softmax_cross_entropy_with_logits(labels=p,logits=q,dim=1)

    def accuracy_measure(self,prob,label):
        prediction = tf.argmax(prob, 1)
        answer = tf.argmax(label, 1)
        equality = tf.equal(prediction, answer)
        acc = tf.reduce_mean(tf.cast(equality, tf.float32))
        
        return acc

    def callbackEmg(self, msg): 
        self.emgData = msg.data
        self.emgSampled = True

    def get_dz(self,dz):
        if dz > 0 :
            return np.random.normal(0, dz, size=(self.n_batch, self.n_domain)).astype(np.float32)
        else :
            return np.zeros((self.n_batch, self.n_domain)).astype(np.float32)


    def enque_thread(self):
        print "enque_thread : start"

        self.buf=[]
        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                if self.emgSampled == True :

                    a = np.array( self.emgData )
                    
                    self.sess.run(self.enque_op, feed_dict={ self.a_enque_ph : a,
                                                             self.d_enque_ph : self.current_d,
                                                             self.y_enque_ph : np.zeros([self.n_class])} )
                    self.buf.append(np.concatenate([a,self.current_d]))

                    if len(self.buf) > self.n_buf:
                        self.buf.pop(0)

                    if self.n_ref > 0:
                        for i in range(9): 
                            ref_idx = np.random.choice(self.n_ref,1)[0]

                            self.sess.run(self.enque_op, feed_dict={ self.a_enque_ph : self.ref_target_X[ref_idx],
                                                                    self.d_enque_ph : self.current_d,
                                                                    self.y_enque_ph : self.ref_target_Y[ref_idx]} )

                    
                    self.buf_flag = True
                    self.emgSampled = False

                time.sleep(0.001)
        
    def get_dz(self, sigma):
        if sigma > 0:
            return np.random.normal(0, sigma, size=(self.n_batch, self.n_domain)).astype(np.float32)
        else :
            return np.zeros(shape=(self.n_batch, self.n_domain), dtype=np.float32)


    def train_thread(self):
        print "train_thread : start"

        with self.coord.stop_on_exception():

            acc_sum = 0.
            lossD_sum = 0.
            lossG_sum = 0.
            epoch_max = 5
            epoch = 0
            step_max = 10000
            step_max_global = (epoch_max-1)*step_max
            step = 0
            while not self.coord.should_stop():

                step_global = epoch*step_max+step
                dz_step = (step_max_global-step_global)*self.sigma_dz/step_max_global

                if args.ros == True:
                    self.loop_rate.sleep()

                    size = self.sess.run(self.que_size)

                    if size < self.n_batch*10 :
                        continue
                    # Sample Target domain from queue
                    Xt, Yt = self.sess.run(self.deque_op)
                else :
                    time.sleep(0.001)

                    dz = self.get_dz(dz_step) 
                    [Xt,Yt] = self.sess.run( self.sample_batch_target, feed_dict={self.d_target : self.current_d_batch , self.dz : dz} )

                    #print Xt

                # Sample Normal domain
                dz = self.get_dz(0)
                [Xn, Yn] = self.sess.run( self.sample_batch_normal, feed_dict={self.dz : dz} )
                # Sample Source domain
                dz = self.get_dz(dz_step) 
                [Xs, Ys] = self.sess.run( self.sample_batch_source, feed_dict={self.dz : dz} )

                
                feed_dict = { self.X_normal : Xn, self.Y_normal : Yn, 
                              self.X_source : Xs, self.Y_source : Ys , 
                              self.X_target : Xt ,self.Y_target : Yt ,
                              self.phase : True, self.keep_prob : 0.8}
                 
                if self.use_z == True :
                    z_val = np.random.normal(0, 1, size=(self.n_batch, self.n_z)).astype(np.float32)
                    feed_dict[self.z_prior] = z_val
                    
                # Train D
                _, lossD, acc_n = self.sess.run( [self.train_D, self.loss_D, self.acc_n] , feed_dict=feed_dict)
                lossD_sum = lossD_sum + float(lossD)

                # Train G
                _, lossG, acc_s = self.sess.run( [self.train_G, self.loss_G, self.acc_s] , feed_dict=feed_dict)
                lossG_sum = lossG_sum + float(lossG)

                acc_sum = acc_sum + float(acc_n)     
                

                if (step+1)%100 == 0 :
                    print "epoch : {:d}    step : {:d}    lossD : {:f}    lossG : {:f}   accuracy : {:.2f}".format(epoch,step+1,lossD_sum/100,lossG_sum/100,acc_sum)
                    acc_sum = 0.
                    lossD_sum = 0.
                    lossG_sum = 0.

                    
                    self.sess.run(self.update_network_params)

                    summary_str = self.sess.run( self.summary_train, feed_dict = feed_dict )
                    self.summary_writer.add_summary( summary_str, step_max*epoch + step )
                    self.summary_writer.flush()

  
                    if (step+1)%1000 == 0 :
                        if args.save == SAVE_ALL :
                            self.gan_saver.save(self.sess,self.gan_model_dir)

                step = step+1
                
                if step >= step_max :
                    step = 0
                    epoch = epoch+1

                    if epoch >= epoch_max :
                        break
                
            # print "Training Done!\n"

            # if args.ros != True:
            #     # Save result for t-sne
            #     savedir = "/home/taeho/catkin_ws/src/bhand/src/tf_data/embedding"
            #     f_normal = open(savedir+'/normal.tsv','w')
            #     f_source = open(savedir+'/source.tsv','w')
            #     f_source_t = open(savedir+'/source_t.tsv','w')
            #     f_target = open(savedir+'/target.tsv','w')

            #     f_normal.write("name\tlabel\tA0\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tPlacement\n")
            #     f_source.write("name\tlabel\tA0\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tPlacement\n")
            #     f_source_t.write("name\tlabel\tA0\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tPlacement\n")
            #     f_target.write("name\tlabel\tA0\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tPlacement\n")
            #     idx = 1
            #     posture_dict={1 : 'rest', 2 : 'pinch', 3 : 'fist' , 4 : 'hook', 5 : 'pointing', 6 : 'one' , 7 : 'two', 8 : 'three', 9 : 'four' , 10 : 'five'}

            #     for i in range(40):
            #         dz = np.random.normal(0, self.sigma_dz, size=(self.n_batch, self.n_domain)).astype(np.float32)
            #         [Xt,Yt] = self.sess.run( self.sample_batch_target, feed_dict={self.d_target : self.current_d_batch , self.dz : dz} )
            #         dz = np.zeros(shape=(self.n_batch, self.n_domain), dtype=np.float32)
            #         [Xn, Yn] = self.sess.run( self.sample_batch_normal, feed_dict={self.dz : dz} )
            #         dz = np.random.normal(0, self.sigma_dz, size=(self.n_batch, self.n_domain)).astype(np.float32)
            #         [Xs, Ys] = self.sess.run( self.sample_batch_source, feed_dict={self.dz : dz} )

            #         feed_dict = { self.X_normal : Xn, self.Y_normal : Yn, 
            #                     self.X_source : Xs, self.Y_source : Ys , 
            #                     self.X_target : Xt ,self.Y_target : Yt ,
            #                     self.phase : True}

            #         Ndecode = self.sess.run(self.decode_normal, feed_dict=feed_dict)        
            #         Sdecode = self.sess.run(self.decode_source, feed_dict=feed_dict)        
            #         Gdecode = self.sess.run(self.decode_source_t, feed_dict=feed_dict)        
            #         Tdecode = self.sess.run(self.decode_target, feed_dict=feed_dict)        

            #         for j in range( self.n_batch ):
            #             f_normal.write("{:d}\t{:d} : ".format(idx,int(Ndecode[j,0])-1) + posture_dict[int(Ndecode[j,0])] + "\t")
            #             f_source.write("{:d}\t{:d} : ".format(idx,int(Sdecode[j,0])-1) + posture_dict[int(Sdecode[j,0])] + "\t")
            #             f_source_t.write("{:d}\t{:d} : ".format(idx,int(Gdecode[j,0])-1) + posture_dict[int(Gdecode[j,0])] + "\t")
            #             f_target.write("{:d}\t{:d} : ".format(idx,int(Tdecode[j,0])-1) + posture_dict[int(Tdecode[j,0])] + "\t")
            #             # f_normal.write("{:d}\t{:d}\t".format(idx, int(Ndecode[j,0])))
            #             # f_source.write("{:d}\t{:d}\t".format(idx, int(Sdecode[j,0])))
            #             # f_source_t.write("{:d}\t{:d}\t".format(idx, int(Gdecode[j,0])))
            #             # f_target.write("{:d}\t{:d}\t".format(idx, int(Tdecode[j,0])))

            #             idx += 1

            #             for data in Ndecode[j,1:9]:
            #                 f_normal.write("{:f}\t".format(data))
            #             for data in Sdecode[j,1:9]:
            #                 f_source.write("{:f}\t".format(data))
            #             for data in Gdecode[j,1:9]:
            #                 f_source_t.write("{:f}\t".format(data)) 
            #             for data in Tdecode[j,1:9]:
            #                 f_target.write("{:f}\t".format(data)) 

            #             f_normal.write("{:f}\n".format(Ndecode[j,9]))
            #             f_source.write("{:f}\n".format(Sdecode[j,9]))
            #             f_source_t.write("{:f}\n".format(Gdecode[j,9]))
            #             f_target.write("{:f}\n".format(Tdecode[j,9]))

            #     f_normal.close()
            #     f_source.close()
            #     f_source_t.close()
            #     f_target.close()

            self.train_done = True
                    

    def predict_thread(self): 
        print "prediction_thread : start"

        with self.coord.stop_on_exception():
            f = open('result.csv','w')
            self.sess.run(self.update_network_params) 
            step = 0
            while not self.coord.should_stop():
                if args.ros == True:
                    self.loop_rate.sleep()

                    if len(self.buf) < self.n_buf or self.buf_flag == False :
                        continue
                    
                    self.buf_flag = False

                    Xt = np.array(self.buf)
                    Yt = np.zeros([self.n_buf,self.n_class])
                else :
                    time.sleep(0.001)
                    # dz_step = 0.001
                    # dz = np.random.normal(0, dz_step, size=(self.n_batch, self.n_domain)).astype(np.float32)
                    dz = self.get_dz(0.0)
                    [Xt,Yt] = self.sess.run( self.sample_batch_target, feed_dict={ self.d_target : self.current_d_batch, self.dz : dz} )

                if self.use_z == True :
                    z_val = np.random.normal(0, 1, size=(self.n_buf, self.n_z)).astype(np.float32)
                    feed_dict[self.z_prior] = z_val

                Dc, G = self.sess.run([self.extract_posture, self.data_transferred], feed_dict={self.X_target:Xt, self.phase : False, self.keep_prob : 1.0})
                posture = Dc

                if (step+1)%10 == 0 :
                    print "class : {}".format(posture)

                    for idx in range(self.n_buf):
                        f.write("{},{},{}\n".format(Xt[idx],G[idx],posture))

                    if args.ros == True:
                        self.pubPredict.publish(posture)
                
                step = step+1
            f.close()

    def estimate_d(self):
        step_max = 1000 #int(args.estimate*250 / self.n_batch)
        step = 0

        while step < step_max :

            if args.ros == True :
                self.loop_rate.sleep()
                size = self.sess.run(self.que_size)

                if size < self.n_batch :
                    continue
                # Sample Target domain from queue
                Xt = self.sess.run(self.deque_op)
                Yt = np.zeros([self.n_batch,self.n_class])
            else :
                time.sleep(0.001)
                [Xt,Yt] = self.sess.run( self.sample_batch_target, feed_dict={ self.d_target : self.current_d_batch })

            feed_dict={self.a_target : Xt[:,:self.n_emg], self.keep_prob : 1.0, self.phase : False}

            if self.use_z == True :
                z_val = np.random.normal(0, 1, size=(self.n_batch, self.n_z)).astype(np.float32)
                feed_dict[self.z_prior] = z_val

            _, loss_d, estimated_d = self.sess.run([self.train_d, self.loss_d, self.d_target_e1], feed_dict=feed_dict) 

            if (step+1)%10 == 0 :
                print "step : {:d}    loss : {:f}".format(step+1,loss_d)
                print estimated_d
            
            step = step+1

        print "Estimated : {}".format(estimated_d)

        self.current_d = estimated_d
        self.current_d_batch = [ estimated_d for _ in range(self.n_batch) ]
            

    def main(self):

        threads = [ threading.Thread(target=self.enque_thread) ,
                    threading.Thread(target=self.train_thread) ,
                    threading.Thread(target=self.predict_thread)] + self.que_thread   

        self.coord.register_thread( threads[0] )
        self.coord.register_thread( threads[1] )
        self.coord.register_thread( threads[2] )
        self.coord.register_thread( threads[3] )
       # self.coord.register_thread( threads[4] )

        # Using ROS : get data from ros messages 
        if args.ros == True :
            #Start ros
            
            rate = rospy.Rate(1000)
        
            threads[0].start() #Enque thread
            if args.mode == MODE_TRAINING:
                threads[1].start() #Train thread
            
            if args.mode == MODE_PREDICTION:
                threads[2].start() #Predict thread

            #Run
            with self.coord.stop_on_exception():
                while not rospy.is_shutdown() and not self.coord.should_stop():
                    rate.sleep()
        
        else : # non-ROS : get data from files

            if args.mode == MODE_TRAINING:
                threads[1].start() #Train thread

            if args.mode == MODE_PREDICTION:
                threads[2].start() #Predict thread

            #Run
            with self.coord.stop_on_exception():
                while not self.coord.should_stop():
                    if self.train_done == True:
                        break
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
            help="(INT) Mode configuration [ 0: prediction(default) , 1: training ]"
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
            help="(INT) Load model( Do not Load : 0(defalut), Load all : 1 )"
            )

    parser.add_argument(
            '-save',
            type=int,
            default=0,
            help="(INT) Save model( Do not Save : 0(default), Save all : 1 )"
            )
    parser.add_argument(
            '-estimate',
            type=int,
            default=0,
            help="(INT) Estimate placement measure( OFF : 0(default), ON : estimating time in second )"
    )

    args, unparsed = parser.parse_known_args()

    tensor = RosTF()
    tensor.main()
