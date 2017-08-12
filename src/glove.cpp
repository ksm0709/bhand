#include <stdio.h>

#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"

int main(int argc, char **argv)
{
	ros::init(argc,argv,"glove_node");
	ros::NodeHandle nh;

	ros::Publisher msg_glove_pub = nh.advertise<std_msgs::Float32MultiArray>("msg_glove",10);
	ros::Rate loop_rate(10);
	int count = 0;

	ROS_INFO("GLOVE_NODE Started!\n");

	while( ros::ok() )
	{
		std_msgs::Float32MultiArray msg;
		
		/** Get Data from glove **/

		/*************************/

		msg_glove_pub.publish(msg);
		loop_rate.sleep();
	}

	return 0;
}
