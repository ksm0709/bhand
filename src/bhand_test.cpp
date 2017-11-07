#include "ros/ros.h"
#include "bhand/EmgArray.h"

void callback_emg(const bhand::EmgArray::ConstPtr& arr);

int16_t emgData[16];
bool emgReady;

int main(int argc, char **argv)
{
	ros::init(argc,argv,"resWatch");
	ros::NodeHandle nh;

	ros::Subscriber sub_myo = nh.subscribe<bhand::EmgArray>("myo_emg",1,callback_emg);
	ros::Rate loop_rate(300);

	FILE *fp = fopen("data.txt","w");
	emgReady = false;

	while( ros::ok() )
	{
		if( emgReady )
		{
			fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d\n", emgData[0],emgData[1],emgData[2],emgData[3],emgData[4],emgData[5],emgData[6],emgData[7],rcp);
			emgReady = false;
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

