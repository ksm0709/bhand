#include <ros/ros.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/Int8.h>
#include "robolimb/RobolimbState.h"
#include <stdlib.h>
#include <vector>

#define EXTENSION	0
#define INDEXPOINT	1
#define FISTGRIP	2
#define TRIPODGRIP	3

#define THUMB		0
#define INDEX		1
#define	MIDDLE		2
#define RING		3
#define LITTLE		4
#define ROTATOR		5

#define MODE_QUICK	0
#define MODE_PWM	1

robolimb::RobolimbState hand;
std_msgs::Int16MultiArray cmd;
bool subFlag = false;
int posture; 
int count[5];
int posture_num=0;

void posture_callback(const std_msgs::Int8::ConstPtr& msg);
int max_index(int* arr, int size);
void publish( ros::Publisher* pub, int16_t mode, int16_t p1, int16_t p2, int16_t p3, int16_t p4, int16_t p5, int16_t p6);


int main(int argc, char** argv)
{
	ros::init(argc,argv,"robolimb_controller");
	ros::NodeHandle nh;

//	ros::Subscriber sub_state = nh.subscribe("robolimb_state",10,callback);
	ros::Subscriber sub_posture = nh.subscribe("posture",10,posture_callback);
	ros::Publisher pub_cmd = nh.advertise<std_msgs::Int16MultiArray>("robolimb_cmd",1);

	ros::Rate loop_rate(1);
	
	if( argc > 1 )
	{
		publish( &pub_cmd,  atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
		return 0;
	}


	ROS_INFO("Robolibm control start!");
	while( ros::ok() )
	{
		if( subFlag )
		{
			if( posture_num < 3 )
			{
				count[ posture ]++;		 	
				posture_num++;
			}	
			else
			{
				int idx = max_index( count , 4 );
				ROS_INFO("Execute Posture %d", idx+1 );
				switch( idx	)
				{
				case EXTENSION:
					
					publish( &pub_cmd, MODE_QUICK, 0, 0, 0, 0, 0, 0 );
					ros::Duration(0.5).sleep();
					publish( &pub_cmd,  MODE_PWM, -150, -150, -150, -150, -150, -150);

					break;
				case INDEXPOINT:

					publish( &pub_cmd, MODE_QUICK, 0, 0, 0, 0, 0, 1 );
					ros::Duration(0.5).sleep();
					publish( &pub_cmd, MODE_PWM, 150, 0, 0, 0, 0, 0 );
					
					break;
				case FISTGRIP:
				
					publish( &pub_cmd, MODE_QUICK, 0, 0, 0, 0, 0, 5 );
					ros::Duration(0.5).sleep();
					publish( &pub_cmd, MODE_PWM, 150, 0, 0, 0, 0, 0 );

					break;
				case TRIPODGRIP:

					publish( &pub_cmd, MODE_QUICK, 0, 0, 0, 0, 0, 2 );
					ros::Duration(0.5).sleep();
					publish( &pub_cmd, MODE_PWM, 150, 150, 150, 0, 0, 0 );

					break;
				default:

					break;
				}
				
				memset( count, 0, sizeof(int)*5 );
				posture_num = 0;
				subFlag = false;
			}
		}

		loop_rate.sleep();
		ros::spinOnce();
	}
}


void posture_callback(const std_msgs::Int8 ::ConstPtr& msg)
{
	posture = msg->data;	
	subFlag = true;
}

int max_index(int* arr,int size)
{
	int i,k;

	k = 0;
	for( i = 0; i < size; i++)
		if( arr[ k ] < arr[ i ] )
			k = i;

	return k;
}

void publish( ros::Publisher* pub, int16_t mode, int16_t p1, int16_t p2, int16_t p3, int16_t p4, int16_t p5, int16_t p6)
{
	cmd.data.clear();

	cmd.data.push_back( mode );
	cmd.data.push_back( p1 );
	cmd.data.push_back( p2 );
	cmd.data.push_back( p3 );
	cmd.data.push_back( p4 );
	cmd.data.push_back( p5 );
	cmd.data.push_back( p6 );

	pub->publish( cmd );
}
