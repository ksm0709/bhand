#include "ros/ros.h"
#include "std_msgs/Int8.h"
#include "bhand/EmgArray.h"

void callback_emg(const bhand::EmgArray::ConstPtr& arr);
void callback_rcp(const std_msgs::Int8::ConstPtr& ptr);

int16_t emgData[16];
int8_t rcp;
bool emgReady, rcpReady;

int main(int argc, char **argv)
{
	ros::init(argc,argv,"resWatch");
	ros::NodeHandle nh;

	ros::Subscriber sub1 = nh.subscribe<std_msgs::Int8>("rcpGLOVE",1,callback_rcp);
	ros::Subscriber sub2 = nh.subscribe<bhand::EmgArray>("myo_emg",1,callback_emg);
	ros::Rate loop_rate(100);

	FILE *fp = fopen("data.txt","w");
	emgReady = false;
	rcpReady = false;

	while( ros::ok() )
	{
		if( emgReady && rcpReady )
		{
			fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d\n", emgData[0],emgData[1],emgData[2],emgData[3],emgData[4],emgData[5],emgData[6],emgData[7],rcp);
			emgReady = false;
			rcpReady = false;
		}

		loop_rate.sleep();
		ros::spinOnce();
	}

	fclose(fp);
	return 0;
}

void callback_emg(const bhand::EmgArray::ConstPtr& arr)
{
	int i = 0;

	for(std::vector<int16_t>::const_iterator it = arr->data.begin(); it != arr->data.end(); ++it )
	{
		emgData[i] = *it;
		i++;
	}
	emgReady = true;
	return;
}

void callback_rcp(const std_msgs::Int8::ConstPtr& ptr)
{
	rcp = ptr->data;
	rcpReady = true;
	return;
}
