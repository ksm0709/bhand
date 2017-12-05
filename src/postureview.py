#!/usr/bin/env python  

import rospy
import pylab as plt
import numpy as np

from std_msgs.msg import Float32

time = 0.
posture = 0
flag_t = True
flag_p = True

def callback_time(msg):
    global flag_t, time
    time = msg.data
    flag_t = True

def callback_posture(msg):
    global flag_p, posture
    posture = int(msg.data)
    flag_p = True

if __name__ == '__main__':

    rospy.init_node('postureview')
    rate = rospy.Rate(30)

    subTime = rospy.Subscriber('timePosture', Float32, callback_time, queue_size=1) 
    subTime = rospy.Subscriber('Posture', Float32, callback_posture, queue_size=1) 

    print "Posture Viewer Start"

    path = '/home/taeho/catkin_ws/src/bhand/src/'

    imgs = { 0: plt.imread(path+'postures/a.png'),
             1: plt.imread(path+'postures/b.png'),
             2: plt.imread(path+'postures/c.png'),
             3: plt.imread(path+'postures/d.png'),
             4: plt.imread(path+'postures/e.png'),
             5: plt.imread(path+'postures/f.png'),
             6: plt.imread(path+'postures/g.png'),
             7: plt.imread(path+'postures/h.png'),
             8: plt.imread(path+'postures/i.png'),
             9: plt.imread(path+'postures/j.png'),
             19: plt.imread(path+'postures/norm.png'),
             20: plt.imread(path+'postures/white.png')}

    plt.ion()
    fig = plt.figure(1, facecolor='white')
    img = None
    txt = None
    txt_ = None

    p1 = fig.add_subplot(121)
    p2 = fig.add_subplot(224)
    p3 = fig.add_subplot(222)
    p1.set_title('Current')
    p2.set_title('Next')
    p1.axis('off')
    p2.axis('off')
    p3.axis('off')
    while not rospy.is_shutdown() :
        
        if flag_t == True : 
            if txt is None :
                txt_= p3.text(10,120,'Remaining Time',fontsize=14)
                txt = p3.text(10,160,'{:.1f}'.format(time),fontsize=16,style='italic',color='red')
            else :
                txt.set_text('{:.1f}'.format(time))
            flag_t = False

        if flag_p == True :
            if img is None :
                img = p1.imshow( imgs[posture] )
                img_next = p2.imshow(imgs[(posture+1)%10])
                p3.imshow( imgs[20] )
            else :
                img.set_data( imgs[posture] )
                img_next.set_data( imgs[(posture+1)%10] )
            flag_p = False
        
        plt.pause(.01)
        plt.draw()
                
        rate.sleep()
    
    