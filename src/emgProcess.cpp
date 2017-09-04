#include "ros/ros.h"

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "bhand/EmgArray.h"
#include "std_msgs/Float32MultiArray.h"
#include <stdlib.h>
#include "dsp.h"
#include "MuscleBase.h"

#define USE_MYO		1

#define MODE_RUN	0
#define MODE_MIN	1
#define MODE_MAX	2
#define channel_MAX		32

MuscleBase muscle;

int channel;

float resData[channel_MAX];
float mavData[channel_MAX];
float act[channel_MAX];
float rawData[channel_MAX];
float max[channel_MAX];
float min[channel_MAX];
bool readFlag = false;

void callback_myo_emg(const bhand::EmgArray::ConstPtr& arr);
void callback_rawEMG(const std_msgs::Float32MultiArray::ConstPtr& arr);
void saveParam(ros::NodeHandle &nh);
void showParam(void);

int main(int argc, char **argv)
{
	FILE *datafp = fopen("emgdata.txt","w");
	char mode = -1;

	if( argc )
	{
		switch( argv[1][1] )
		{
			case 'l': mode = MODE_MIN; break;
			case 'h': mode = MODE_MAX; break;
			case 'r': mode = MODE_RUN; break;
			default: printf("\n  Mode Error!\n\t-l : MODE_MIN\n\t-h : MODE_MAX \n\t-r : MODE_RUN \n\n");
					 return 0;
					 break;
		}

		if( argc > 2 )
			channel = atoi(argv[2]);	
	}


	ros::init(argc,argv,"emgProcess");
	ros::NodeHandle nh;

#if USE_MYO == 1
	ros::Subscriber sub = nh.subscribe("myo_emg", 1, callback_myo_emg); 
	channel = 8;
#else
	ros::Subscriber sub = nh.subscribe("rawEMG",10, callback_rawEMG);
	channel = 8;
#endif

	ros::Publisher pub = nh.advertise<std_msgs::Float32MultiArray>("actEMG",10);
//	ros::Publisher pubf = nh.advertise<bhand::EmgArray>("firEMG",10);

	ros::Rate loop_rate(500);
	
	std::vector<float> list;
	bhand::EmgArray fir_msg;	
	std_msgs::Float32MultiArray msg;
	float tempAct;
	float tempRms;
	int i,save_count=0;

	firFilter_init();
	rmsFilter_init(64);

	system("rosparam load ~/catkin_ws/src/bhand/config/emgConfig.yaml");
	
	nh.getParam("Emg/max", list);
	for(i=0;i<list.size();i++)
		max[i] = (float)list[i];

	nh.getParam("Emg/min", list);
	for(i=0;i<list.size();i++)
		min[i] = (float)list[i];	

	ROS_INFO("emgProcess node start %s",argv[1]);
	
	switch( mode )
	{
		case MODE_RUN:
			
			ROS_INFO("mode : MODE_RUN");

			showParam();
			
			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<channel;i++)
					{
						tempAct = (float)(abs(rawData[i])-min[i])/(max[i]-min[i]);
					
						if( tempAct > 1 ) 
							tempAct = 1;
						else if( tempAct < 0 )
							tempAct = 0;

						act[i] = muscle.get_next_activation( act[i] , tempAct, 0.002);
						resData[i] = (float)(rmsFilter_update(i,act[i]*100));

						// For mav filter test
//						mavData[i] = (int8_t)(rmsFilter_update(i,tempAct*100));
					}

					msg.data.clear();
					msg.data.assign( resData, resData + channel );
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
					for(i=0;i<channel;i++)
					{
//						rawData[i] = firFilter_update(i, rawData[i]);
						tempRms = rmsFilter_update(i, abs(rawData[i]));

						if( tempRms > max[i] )
							max[i] = tempRms;
					}

					readFlag = false;
				}
				
				save_count++;

				if( save_count % 50 == 0 )
				{
					showParam();

					for(i=0;i<list.size();i++)
						list[i] = max[i];
					nh.setParam("bhand/Emg/max", list);

					save_count = 0;
				}

				ros::spinOnce();
				loop_rate.sleep();	
			}

			system("rosparam dump ~/catkin_ws/src/bhand/config/emgConfig.yaml bhand");

			ROS_INFO("DONE!");
		
			break;

		case MODE_MIN:
		
			ROS_INFO("mode : MODE_MIN");

			for(i=0;i<list.size();i++)
				min[i] = 0;

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<channel;i++)
					{
//						rawData[i] = firFilter_update(i, rawData[i]);
						tempRms = rmsFilter_update(i, abs(rawData[i]));

						if( tempRms > min[i] )
							min[i] = tempRms;
					}

					readFlag = false;
				}
		
				save_count++;

				if( save_count % 50  == 0)
				{
					showParam();

					for(i=0;i<list.size();i++)
						list[i] = min[i];
					nh.setParam("bhand/Emg/min", list);

					save_count = 0;
				}	

				ros::spinOnce();
				loop_rate.sleep();
			}
			
			system("rosparam dump ~/catkin_ws/src/bhand/config/emgConfig.yaml bhand");

			ROS_INFO("DONE!");

		break;
	}

	fclose(datafp);	
	return 0;
}

void callback_myo_emg(const bhand::EmgArray::ConstPtr& arr)
{
	int i = 0;

	for(std::vector<int16_t>::const_iterator it = arr->data.begin(); it != arr->data.end(); ++it )
	{
		rawData[i] = (float)*it;
		i++;
	}

	readFlag = true;
	return;
}

void callback_rawEMG(const std_msgs::Float32MultiArray::ConstPtr& arr)
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

	printf("Emg:\n");
	printf("    max: [");
	for(i=0;i<channel;i++)
	{
		printf("%.1f",max[i]);
		if( i < channel-1 )
			printf(",");
		else
			printf("]\n");
	}	

	printf("    min: [");
	for(i=0;i<channel;i++)
	{
		printf("%.1f",min[i]);
		if( i < channel-1 )
			printf(",");
		else
			printf("]\n");
	}

	printf("\n");
}
