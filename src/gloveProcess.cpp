#include "ros/ros.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int8MultiArray.h"
#include "std_msgs/Int8.h"
#include "dsp.h"


#define MODE_RUN	0
#define MODE_MIN	1
#define MODE_MAX	2

int8_t resData[5];
double rawData[5];
int max[5];
int min[5];
bool readFlag = false;

void callback(const std_msgs::Int8MultiArray::ConstPtr& arr);
void saveParam(void);

int main(int argc, char **argv)
{
	char mode = -1;

	if( argc )
		switch( argv[1][1] )
		{
			case 'l': mode = MODE_MIN; break;
			case 'h': mode = MODE_MAX; break;
			case 'r': mode = MODE_RUN; break;
			default: printf("\n  Mode Error!\n\t-l : MODE_MIN\n\t-h : MODE_MAX \n\t-r : MODE_RUN \n\n");
					 return 0;
					 break;
		}

	ros::init(argc,argv,"gloveProcess");
	ros::NodeHandle nh;

	ros::Subscriber sub = nh.subscribe("rawGLOVE", 1, callback); 
	ros::Publisher pub = nh.advertise<std_msgs::Int8MultiArray>("normGLOVE",10);
	ros::Publisher pub_rcp = nh.advertise<std_msgs::Int8>("rcpGLOVE",10);

	ros::Rate loop_rate(200);

	std_msgs::Int8 msg_rcp;
	std_msgs::Int8MultiArray msg;
	std::vector<int> list;
	double temp;
	int i;

	firFilter_init();
	rmsFilter_init(32);

	system("rosparam load ~/catkin_ws/src/bhand/config/gloveConfig.yaml");

	nh.getParam("Glove/max", list);
	for(i=0;i<list.size();i++)
		max[i] = list[i];

	nh.getParam("Glove/min", list);
	for(i=0;i<list.size();i++)
		min[i] = list[i];	

	ROS_INFO("gloveProcess node start %s",argv[1]);

	int t[5];

	switch( mode )
	{
		case MODE_RUN:

			ROS_INFO("mode : MODE_RUN");

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<5;i++)
					{
						temp = (double)(rawData[i]-min[i])/(max[i]-min[i]);

						if( temp > 1 ) 
							temp = 1;
						else if( temp < 0 )
							temp = 0;

						resData[i] = (int8_t)(temp*100);

						if( resData[i] > 80 )
							t[i] = 1;
						else
							t[i] = 0;
					}

					// rcp
					int st = t[0]+t[1]+t[2]+t[3]+t[4];

					if( st == 5 )
					{
						msg_rcp.data = 0;	
					}
					else if( st == 3 )
					{
						msg_rcp.data = 1;
					}
					else if( st == 0 ){
						msg_rcp.data = 2;
					}
					else
						msg_rcp.data = 3;

					// finger
//					if( t[0] )
//					{
//						msg_rcp.data = 0;
//					}
//					else if( t[1] || t[2] )
//					{
//						msg_rcp.data = 1;
//					}
//					else if( t[3] || t[4] )
//					{
//						msg_rcp.data = 2;
//					}
//					else
//						msg_rcp.data = 3;

					pub_rcp.publish( msg_rcp );

					msg.data.clear();
					msg.data.assign( resData, resData + 5 );
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
					for(i=0;i<5;i++)
						if( rawData[i] > max[i] )
							max[i] = rawData[i];

					readFlag = false;
				}

				ros::spinOnce();
				loop_rate.sleep();	

			}

			for(i=0;i<list.size();i++)
				list[i] = max[i];
			nh.setParam("Glove/max", list);
			saveParam();

			break;

		case MODE_MIN:

			ROS_INFO("mode : MODE_MIN");

			for(i=0;i<list.size();i++)
				min[i] = 0;

			while(ros::ok())
			{
				if( readFlag )
				{
					for(i=0;i<5;i++)
						if( rawData[i] > min[i] )
							min[i] = rawData[i];

					readFlag = false;
				}

				ros::spinOnce();
				loop_rate.sleep();	

			}

			for(i=0;i<list.size();i++)
				list[i] = min[i];
			nh.setParam("Glove/min", list);
			saveParam();

			break;
	}


	return 0;
}

void callback(const std_msgs::Int8MultiArray::ConstPtr& arr)
{
	int i = 0;

	for(std::vector<int8_t>::const_iterator it = arr->data.begin(); it != arr->data.end(); ++it )
	{
		rawData[i] = *it;
		i++;
	}

	readFlag = true;
	return;
}

void saveParam(void)
{
	FILE *fp = fopen("~/catkin_ws/src/bhand/config/gloveConfig.yaml","w");
	int i;

	fprintf(fp,"Glove:\n");
	fprintf(fp,"    max: [");
	for(i=0;i<5;i++)
	{
		fprintf(fp,"%d",max[i]);
		if( i < 4 )
			fprintf(fp,",");
		else
			fprintf(fp,"]\n");
	}	

	fprintf(fp,"    min: [");
	for(i=0;i<5;i++)
	{
		fprintf(fp,"%d",min[i]);
		if( i < 4 )
			fprintf(fp,",");
		else
			fprintf(fp,"]\n");
	}	

	fclose(fp);

}
