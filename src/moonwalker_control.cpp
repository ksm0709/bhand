#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include "moonwalker.h"

#include <stdlib.h>

#define MWK_NUM 	1
#define MOTOR_NUM   MWK_NUM*2

float desired[MWK_NUM*2];
int mwk_num;
long mwk_id_list[MWK_NUM] = { MWK1_ID }; 
long motor_id_list[MWK_NUM*2];
bool flag[MOTOR_NUM];

void des_callback(const std_msgs::Float32MultiArray::ConstPtr& arr);

int main(int argc, char** argv)
{
	ros::init(argc,argv,"moonwalker_control");
	ros::NodeHandle nh;

	ros::Subscriber sub_desPos = nh.subscribe("moonwalker_des",10,des_callback);
//	ros::Publisher pub_cmd = nh.advertise<std_msgs::Float32MultiArray>("",1);

	ros::Rate loop_rate(10);
	Moonwalker moonwalker;

	int i;
	int channel=0;
	if( argc >1 )
		channel = atoi(argv[1]);

	moonwalker.init(channel);

	for(i=0;i<MWK_NUM;i++)
	{
		motor_id_list[i*2] = mwk_id_list[i]; 
		motor_id_list[i*2+1] = mwk_id_list[i]; 
	}
		
	for(i=0;i<MWK_NUM;i++)
	{
		ROS_INFO("[DRIVER %d | ID %d] Checking Connection ... ",i+1,mwk_id_list[i]);

		moonwalker.Tx( OBJ_READ, mwk_id_list[i], 0, num_motors, 0 ); 

		while( !moonwalker.readFlag )
		{
			moonwalker.Rx();

			loop_rate.sleep();
			ros::spinOnce();
		}
		moonwalker.readFlag = false;

		ROS_INFO("Motor num : %.0f", moonwalker.read(mwk_id_list[i], 0, num_motors));

		moonwalker.Tx( OBJ_WRITE, mwk_id_list[i], 1, command, DRIVER_CMD_CLEAR_FAULT_FLAG );
		moonwalker.Tx( OBJ_WRITE, mwk_id_list[i], 1, command, DRIVER_CMD_MOTOR_ON );
	}
	
	
	while( ros::ok() )
	{
		
		for(i=0;i<MOTOR_NUM;i++)
		{
			if( flag[i] )
			{
				moonwalker.Tx( OBJ_WRITE, motor_id_list[i], i%2+1, current_command, desired[i] );

				ROS_INFO("ID:%ld | Motor %d | Set current to %.2f", motor_id_list[i], i%2+1, desired[i]); 

				flag[i] = false;
			}
		}

		moonwalker.Rx();

		loop_rate.sleep();
		ros::spinOnce();
	}
}

void des_callback(const std_msgs::Float32MultiArray::ConstPtr& arr)
{
	int i = 0;
	
	for( std::vector<float>::const_iterator it = arr->data.begin(); it != arr->data.end(); ++it )
	{
		if( *it != desired[i] )
		{
			desired[i] = *it;
			flag[i] = true;
		}
		else
			flag[i] = false;

		i++;
	}

	return;
}
