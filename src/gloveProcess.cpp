#include "ros/ros.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32MultiArray.h"
#include "dsp.h"

#include "string.h"

#define MODE_RUN	0
#define MODE_MIN	1
#define MODE_MAX	2
#define MODE_SAVE	3

#define SAMPLIG_HZ	100
#define DeltaT		1/SAMPLIG_HZ

#define DATA_SIZE	5
#define STATE_SIZE	5

float resData[STATE_SIZE];
float rawData[DATA_SIZE];
float max[DATA_SIZE];
float min[DATA_SIZE];
bool readFlag = false;

void callback(const std_msgs::Float32MultiArray::ConstPtr& arr);
void showParam(void);

int main(int argc, char **argv)
{
	char mode = -1;
	char filedir[100] = "/home/taeho/catkin_ws/src/bhand/src/tf_data/";
	
	if( argc )
		switch( argv[1][1] )
		{
			case 'l': mode = MODE_MIN; break;
			case 'h': mode = MODE_MAX; break;
			case 'r': mode = MODE_RUN; break;
			case 's': mode = MODE_SAVE;
					  if( argc < 2 )
					  {
						  printf("\n  Filename required!\n\n");
						  return 0;
					  }
					  strcpy(filedir+strlen(filedir),argv[2]);
					  break;
			default: printf("\n  Mode Error!\n\t-l : MODE_MIN\n\t-h : MODE_MAX \n\t-r : MODE_RUN \n\t-s 'filename' : MODE_SAVE\n\n");
					 return 0;
					 break;
		}
	
	
	ros::init(argc,argv,"gloveProcess");
	ros::NodeHandle nh;

	ros::Subscriber sub = nh.subscribe("vmg30_fingers", 1, callback); 
	ros::Publisher pub = nh.advertise<std_msgs::Float32MultiArray>("finger_state",10);

	ros::Rate loop_rate(SAMPLIG_HZ);

	std_msgs::Float32MultiArray msg;
	std::vector<float> list;
	float temp;
	int i, save_count=0;

	rmsFilter_init(32);

	system("rosparam load ~/catkin_ws/src/bhand/config/gloveConfig.yaml");

	nh.getParam("Glove/max", list);
	for(i=0;i<list.size();i++)
		max[i] = list[i];

	nh.getParam("Glove/min", list);
	for(i=0;i<list.size();i++)
		min[i] = list[i];	

	ROS_INFO("gloveProcess node start %s",argv[1]);

	switch( mode )
	{
		case MODE_RUN:

			ROS_INFO("mode : MODE_RUN");
			showParam();

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<DATA_SIZE;i++)
					{
						temp = (float)(rawData[i]-min[i])/(max[i]-min[i]);

						if( temp > 1 ) 
							temp = 1;
						else if( temp < 0 )
							temp = 0;

						temp *= 100;
					
						resData[i+DATA_SIZE] = ( temp - resData[i] ) / DeltaT; // Velocity
						resData[i] = temp;	// Position
					}

					msg.data.clear();
					msg.data.assign( resData, resData + STATE_SIZE );
					pub.publish( msg );					

					readFlag = false;
				}

				ros::spinOnce();
				loop_rate.sleep();	

			}

			break;

		case MODE_MAX:

			ROS_INFO("mode : MODE_MAX");

			for(i=0;i<list.size();i++)
				max[i] = 0;

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<DATA_SIZE;i++)
						if( rawData[i] > max[i] )
							max[i] = rawData[i];

					readFlag = false;
				}

				save_count++;

				if( save_count % 20  == 0)
				{
					showParam();

					for(i=0;i<list.size();i++)
						list[i] = max[i];
					nh.setParam("bhand/Glove/max", list);

					save_count = 0;
				}	

				ros::spinOnce();
				loop_rate.sleep();	

			}

			system("rosparam dump ~/catkin_ws/src/bhand/config/gloveConfig.yaml bhand");

			break;

		case MODE_MIN:

			ROS_INFO("mode : MODE_MIN");

			for(i=0;i<list.size();i++)
				min[i] = 0;

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<DATA_SIZE;i++)
						if( rawData[i] > min[i] )
							min[i] = rawData[i];

					readFlag = false;
				}
				
				save_count++;

				if( save_count % 20  == 0)
				{
					showParam();

					for(i=0;i<list.size();i++)
						list[i] = min[i];
					nh.setParam("bhand/Glove/min", list);

					save_count = 0;
				}
			
				ros::spinOnce();
				loop_rate.sleep();	

			}

			system("rosparam dump ~/catkin_ws/src/bhand/config/gloveConfig.yaml bhand");

			break;

		case MODE_SAVE:
			ROS_INFO("mode : MODE_SAVE");
			FILE* fp = fopen(filedir,"w");

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<DATA_SIZE-1;i++)
						fprintf(fp,"%.0lf,",rawData[i]);
					fprintf(fp,"%.0lf\n",rawData[DATA_SIZE-1]);

					readFlag = false;
				}
				ros::spinOnce();
				loop_rate.sleep();
			}

			fclose(fp);
			break;
	}

	return 0;
}

void callback(const std_msgs::Float32MultiArray::ConstPtr& arr)
{
	int i = 0;

	for(std::vector<float>::const_iterator it = arr->data.begin(); it != arr->data.end(); ++it )
	{
		rawData[i] = *it;
		i++;
	}

	readFlag = true;
	return;
}

void showParam(void)
{
	int i;

	printf("Glove:\n");
	printf("    max: [");
	for(i=0;i<DATA_SIZE;i++)
	{
		printf("%.1f",max[i]);
		if( i < DATA_SIZE-1 )
			printf(",");
		else
			printf("]\n");
	}	

	printf("    min: [");
	for(i=0;i<DATA_SIZE;i++)
	{
		printf("%.1f",min[i]);
		if( i < DATA_SIZE-1 )
			printf(",");
		else
			printf("]\n");
	}

	printf("\n");
}
