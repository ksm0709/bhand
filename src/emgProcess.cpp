#include "ros/ros.h"

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "bhand/EmgArray.h"
#include "std_msgs/Int16MultiArray.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float32.h"
#include <stdlib.h>
#include <queue>
#include "dsp.h"
#include "MuscleBase.h"

#define USE_MYO		1

#define MODE_RUN	0
#define MODE_NORM	1
#define STEP_INIT	0
#define STEP_RUN	1

#define channel_MAX		32
#define SamplingHz		250

class energy
{
private:
	int window_size;
	std::queue<float> que;
	float energy_sum;

public:
	void set(int window_size_)
	{
		window_size = window_size_;
		energy_sum = 0;
	}

	void push(float signal)
	{
		que.push(signal);
		energy_sum += signal*signal;

		if( que.size() > window_size )
		{
			energy_sum -= que.front()*que.front();
			que.pop();
		}
	}

	float get()
	{
		return energy_sum;
	} 
};

class emgProcess
{
private:
	// Normal Variables
	bool readFlag;
	bool save, normalize;
	float resData[channel_MAX];
	float rmsData[channel_MAX];
	float act[channel_MAX];
	float rawData[channel_MAX];
	float max[channel_MAX];
	float min[channel_MAX];
	float tempAct;
	float tempRms;
	float run_time;
	double s_time, c_time;
	char filename[100];
	char mode, step;
	int posture, domain;
	int posture_max, domain_max;
	int channel;

	// Class, STL, FILE
	MuscleBase muscle;
	energy* fir_energy;
	std::vector<float> list;
	FILE *datafp;

	// ROS
	ros::NodeHandle* nh;
	ros::Subscriber sub_emg;
	ros::Publisher pub_act, pub_burst, pub_time, pub_posture;
	std_msgs::Float32MultiArray msg;
	std_msgs::Float32 msg_burst, msg_time, msg_posture;
	
public:

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

	~emgProcess()
	{
		if( fclose )
			fclose(datafp);	
		delete fir_energy;
	}

	emgProcess(ros::NodeHandle* nh_, int channel_, bool save_, bool normalize_) 
		: nh(nh_), channel(channel_), save(save_), normalize(normalize_)
	{
		int i;
		
		system("rosparam load ~/catkin_ws/src/bhand/config/emgConfig.yaml");
		
		nh->getParam("Emg/max", list);
		for(i=0;i<list.size();i++)
			max[i] = (float)list[i];

		nh->getParam("Emg/min", list);
		for(i=0;i<list.size();i++)
			min[i] = (float)list[i];	

		ROS_INFO("emgProcess node start");

		showParam();

		#if USE_MYO == 1
			sub_emg = nh->subscribe("/myo_emg", 1,&emgProcess::callback_myo_emg, this); 
		#else
			sub_emg = nh->subscribe("/rawEMG",10, &emgProcess::callback_rawEMG, this);
		#endif

		pub_act = nh->advertise<std_msgs::Float32MultiArray>("actEMG",1);
		pub_burst = nh->advertise<std_msgs::Float32>("burstEMG",1);
		pub_time = nh->advertise<std_msgs::Float32>("timePosture",1);
		pub_posture = nh->advertise<std_msgs::Float32>("Posture",1);

		fir_energy = new energy[channel];
		for(int i=0;i<channel;i++)
			fir_energy[i].set(16);
		
		s_time = ros::Time::now().toSec();	
		readFlag = false;	
		posture = 0;
		posture_max = 10;
		domain_max = 16;

		if( normalize_ )
		{
			mode = MODE_NORM;
			step = STEP_INIT;
		}
		else
		{
			mode = MODE_RUN;
			step = STEP_INIT;
		}
		
		if( save_ )
		{
			char name[100];
			fflush(stdin);
			printf("Input 'NAME' 'DOMAIN' : ");
			scanf("%s %d",name,&domain);
			
			sprintf(filename,"/home/taeho/catkin_ws/src/bhand/src/tf_data/thesis/%s_d%d.csv",name,domain);
			
			datafp = fopen(filename,"w");
		}
		else
			datafp = NULL;
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

	void modeRun()
	{
		int i;

		if( step == STEP_INIT )
		{
			ROS_INFO("mode : MODE_RUN");

			firFilter_init(channel);
			rmsFilter_init(16);

			showParam();

			step = STEP_RUN;
		}	

		float burst_sum = 0;
		for(i=0;i<channel;i++)
		{
			tempAct = (float)(abs(rawData[i])-min[i])/(max[i]-min[i]);
		
			if( tempAct > 1 ) 
				tempAct = 1;
			else if( tempAct < 0 )
				tempAct = 0;

			act[i] = muscle.get_next_activation( act[i] , tempAct, 0.004);
			resData[i] = (float)(rmsFilter_update(i,act[i]));
			
			fir_energy[i].push( (float)(firFilter_update(i,tempAct)) );
			burst_sum += fir_energy[i].get();
		}
		msg.data.clear();
		msg.data.assign( resData, resData + channel );
		pub_act.publish( msg );					

		msg_burst.data = burst_sum/channel;
		pub_burst.publish( msg_burst );

		if( save )
		{
			// 7초마다 posture 변경
			if( run_time > 8.0 )
			{
				s_time = c_time = 0;
				posture = (posture + 1)%posture_max;

				msg_posture.data = (float)posture;
				pub_posture.publish( msg_posture );
			}
			msg_time.data = 8.0-run_time;
			pub_time.publish( msg_time );

			if( run_time >= 3.0 && run_time < 7.0 && datafp != NULL)
			{
				// EMG 출력
				for(i=0;i<channel;i++)
				{
					fprintf(datafp,"%.6f,",resData[i]);
				}
				// Domain 출력
				for(i=0;i<domain_max;i++)
				{
					if( i == domain )
						fprintf(datafp,"1.0,");
					else
						fprintf(datafp,"0.0,");
				}
				// Posture 출력
				for(i=0;i<posture_max;i++)
				{
					if( i == posture )
						fprintf(datafp,"1.0");
					else
						fprintf(datafp,"0.0");

					if( i < posture_max-1 )
						fprintf(datafp,",");
				}

				fprintf(datafp,"\n");
			}
		}
	}

	void modeNormalize(double sec)
	{
		int i;

		if( step == STEP_INIT )
		{
			ROS_INFO("mode : MODE_NORMALIZE");
			
			rmsFilter_init(64);

			for(i=0;i<list.size();i++)
			{
				max[i] = 0;
				min[i] = 10000;
			}

			step = STEP_RUN;
		}

		if( run_time < sec )
		{
			if( save )
			{
				msg_posture.data = 19;
				pub_posture.publish( msg_posture );
				msg_time.data = (float)sec-run_time;
				pub_time.publish( msg_time );
			}

			for(i=0;i<channel;i++)
			{
				tempRms = rmsFilter_update(i, abs(rawData[i]));

				if( tempRms > max[i] )
					max[i] = tempRms;
				else if( tempRms < min[i] )
					min[i] = tempRms;
			}
			
			// 0.1초 마다 min, max 저장
			if( (int)(run_time*100)%10 == 0 )
			{
				showParam();

				for(i=0;i<list.size();i++)
					list[i] = max[i];
				nh->setParam("bhand/Emg/max", list);

				for(i=0;i<list.size();i++)
					list[i] = min[i];
				nh->setParam("bhand/Emg/min", list);
			}
		}
		else
		{
			step = STEP_INIT;
			mode = MODE_RUN;
			
			s_time = c_time = 0;

			if( save )
			{
				posture = 0;

				msg_posture.data = (float)posture;
				pub_posture.publish( msg_posture );
			}
		}
	}

	void main()
	{
		ros::Rate loop_rate(2*SamplingHz);
		s_time = 0;
		c_time = 0;

		while( ros::ok() )
		{
			ros::spinOnce();
						
			if( !readFlag ) continue;

			readFlag = false;

			c_time += (double)1/SamplingHz;
			run_time = c_time-s_time;

			switch( mode )
			{
				case MODE_RUN: 	modeRun();			break;
				case MODE_NORM: modeNormalize(7);	break;
			}

			loop_rate.sleep();
		}
	}


};


int main(int argc, char **argv)
{
	ros::init(argc, argv, "emgProcess");
	ros::NodeHandle nh;
	
	int channel = 8; 
	bool normalize = false, save = false;

	for(int i = 1 ; i < argc; i++)
	{
		if( !strcmp(argv[i],"-n") )
			normalize = true;	
		else if( !strcmp(argv[i],"-s") )
			save = true;
		else if( !strcmp(argv[i],"-c") )
			channel = atoi(argv[i+1]);
	}

	emgProcess proc(&nh,channel, save, normalize);

	proc.main();

	return 0;
}

