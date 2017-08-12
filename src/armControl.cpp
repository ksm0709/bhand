/*********************************************************
 * Node for bhand arm control ( 2017/6/26 발표 준비용 )
 * 
 * 모드 체인지 & 속도제어 프로토콜 기반
 * 락싸 4CH EMG보드 활용 ( 'emg4board' project )
 *
 * /subscribe : /actEMG
 * /publish : /
 *
 *********************************************************/

#include "ros/ros.h"
#include "std_msgs/Int8MultiArray.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float32MultiArray.h"
#include "moonwalker.h"
#include "serial.h"

#define USE_HW		0
#define USE_HAND	0
#define USE_ARM		1

#define DIR_STOP	0
#define DIR_PLUS	1
#define DIR_MINUS	2

#define CUR_SLOW	0
#define CUR_FAST	1

#define CH1 0
#define CH2	1
#define CH3 2
#define CH4 3
#define THRESHOLD	20
#define TIME_THRESHOLD	10
#define MOVE_THRESHOLD  20

#define MODE_NONE		-1
#define MODE_ELBOW		0
#define MODE_FOREARM	1 
#define MODE_WRIST		2
#define MODE_HAND		3
#define MODE_RESET		4

#define POSTURE_GRASP	0
#define POSTURE_CLICK	1
#define POSTURE_PINCH	2

#define CANID_ELBOW		3
#define CANID_FOREARM	1
#define CANID_WRIST		5

enum{ IDX_SHOULDER,
	  IDX_ELBOW,
	  IDX_FOREARM,
	  IDX_WRIST1,
	  IDX_WRIST2,
	  IDX_THUMB1,
	  IDX_THUMB2,
	  IDX_THUMB3,
	  IDX_INDEX1,
	  IDX_INDEX2,
	  IDX_INDEX3,
	  IDX_MIDDLE1,
	  IDX_MIDDLE2,
	  IDX_MIDDLE3,
	  IDX_SMALL1,
	  IDX_SMALL2,
	  IDX_SMALL3,
	  IDX_LITTLE1,
	  IDX_LITTLE2,
	  IDX_LITTLE3};

#define NUM_JOINT 20

typedef union
{
	struct
	{
		int16_t thumb, index, middle, small, little;
		int16_t in_thumb, in_fingers;
		int8_t	posture;
	} s;

	char bytes[15];
} HandPacket;

int8_t emgData[4];
bool emgReady;

void callback_emg(const std_msgs::Int8MultiArray::ConstPtr& arr);
double abs(double n){ if( n>0 ){ return n; }else{ return -n; }}
int v012( double v ){ if( v>0 ){ return DIR_PLUS; }else if( v<0 ){ return DIR_MINUS; }else{ return DIR_STOP; }}

int main(int argc, char **argv)
{
	ros::init(argc,argv,"armControl");
	ros::NodeHandle nh;

	ros::Subscriber sub1 = nh.subscribe<std_msgs::Int8MultiArray>("actEMG",1,callback_emg);
	ros::Publisher mwk_pub = nh.advertise<std_msgs::Float32MultiArray>("moonwalker_des",1);
	ros::Publisher pub[NUM_JOINT];
	ros::Rate loop_rate(100);
	std_msgs::Float64 pub_msg[NUM_JOINT];
	std_msgs::Float32MultiArray mwk_pub_msg;
	HandPacket handdata;
	Moonwalker arm;
	Serial hand;
	
	int CH3_count = 0;
	int CH3_state_pre, CH3_state_cur;
	int CH3_state_time;
	double desired[4], velocity[4];
	double hand_desired_buf;
	double Ubound[4], Lbound[4];
	double Offset[NUM_JOINT];
	int mode, mode_buf, posture, posture_sel_mode;
	int i, debug_cnt=0;
	bool use_hw = false;


	// For moonwalker current control
	double current_min[3][4] = { 0,0,0,0, 
								 0,0,0,0,		//+
								 0,0,0,0 };		//-
	double current_max[3][4] = { 0,0,0,0,
								 0,0,0,0,
								 0,0,0,0 };

	// [ Motor ][ Direction ][ Speed ] = Current
	double current_map[4][3][2] = { 
	// 	SLOW		FAST
		0,			0,		//Stop	 Elbow 1
		0,			0,		//Pull
		0,			0,		//Undo

		0,			0,		//Stop 	 Elbow 2
		0,			0,		//Pull
		0,			0,		//Undo

		0,			0,		//Stop 	 Forearm 1
		0,			0,		//Pull
		0,			0,		//Undo

		0,			0,		//Stop 	 Forearm 2
		0,			0,		//Pull
		0,			0		//Undo
	};

	if( argc > 1 )
	{
		if( atoi(argv[1]) > 0 )
			use_hw = true;
		else
			use_hw = false;
	}

#if USE_HW == 1
	
	#if USE_HAND == 1
		hand.setHeader(1,"$");
		hand.Open("/dev/ttyUSB1", 115200, 10, 1);
	#endif	
	
	#if USE_ARM == 1
		arm.init(0);

//		arm.Tx( OBJ_WRITE, CANID_ELBOW, 1, command, DRIVER_CMD_CLEAR_FAULT_FLAG );
//		arm.Tx( OBJ_WRITE, CANID_FOREARM, 1, command, DRIVER_CMD_CLEAR_FAULT_FLAG );
//		arm.Tx( OBJ_WRITE, CANID_WRIST, 1, command, DRIVER_CMD_CLEAR_FAULT_FLAG );
//		arm.Tx( OBJ_WRITE, CANID_ELBOW, 2, command, DRIVER_CMD_CLEAR_FAULT_FLAG );
//		arm.Tx( OBJ_WRITE, CANID_FOREARM, 2, command, DRIVER_CMD_CLEAR_FAULT_FLAG );
		arm.Tx( OBJ_WRITE, CANID_WRIST, 2, command, DRIVER_CMD_CLEAR_FAULT_FLAG );


//		arm.Tx( OBJ_WRITE, CANID_ELBOW, 1, command, DRIVER_CMD_MOTOR_ON );
//		arm.Tx( OBJ_WRITE, CANID_FOREARM, 1, command, DRIVER_CMD_MOTOR_ON );
//		arm.Tx( OBJ_WRITE, CANID_WRIST, 1, command, DRIVER_CMD_MOTOR_ON );
//		arm.Tx( OBJ_WRITE, CANID_ELBOW, 2, command, DRIVER_CMD_MOTOR_ON );
//		arm.Tx( OBJ_WRITE, CANID_FOREARM, 2, command, DRIVER_CMD_MOTOR_ON );
		arm.Tx( OBJ_WRITE, CANID_WRIST, 2, command, DRIVER_CMD_MOTOR_ON );
	#endif
#endif

	pub[IDX_SHOULDER] = nh.advertise<std_msgs::Float64>("/rrbot/joint1_position_controller/command", 1);
	pub[IDX_ELBOW] = nh.advertise<std_msgs::Float64>("/rrbot/joint2_position_controller/command", 1);
	pub[IDX_FOREARM] = nh.advertise<std_msgs::Float64>("/rrbot/joint3_position_controller/command", 1);
	pub[IDX_WRIST1] = nh.advertise<std_msgs::Float64>("/rrbot/wrist1_position_controller/command", 1);
	pub[IDX_WRIST2] = nh.advertise<std_msgs::Float64>("/rrbot/wrist2_position_controller/command", 1);
	pub[IDX_THUMB1] = nh.advertise<std_msgs::Float64>("/rrbot/thumb1_position_controller/command", 1);
	pub[IDX_THUMB2] = nh.advertise<std_msgs::Float64>("/rrbot/thumb2_position_controller/command", 1);
	pub[IDX_THUMB3] = nh.advertise<std_msgs::Float64>("/rrbot/thumb3_position_controller/command", 1);
	pub[IDX_INDEX1] = nh.advertise<std_msgs::Float64>("/rrbot/index1_position_controller/command", 1);
	pub[IDX_INDEX2] = nh.advertise<std_msgs::Float64>("/rrbot/index2_position_controller/command", 1);
	pub[IDX_INDEX3] = nh.advertise<std_msgs::Float64>("/rrbot/index3_position_controller/command", 1);
	pub[IDX_MIDDLE1] = nh.advertise<std_msgs::Float64>("/rrbot/middle1_position_controller/command", 1);
	pub[IDX_MIDDLE2] = nh.advertise<std_msgs::Float64>("/rrbot/middle2_position_controller/command", 1);
	pub[IDX_MIDDLE3] = nh.advertise<std_msgs::Float64>("/rrbot/middle3_position_controller/command", 1);
	pub[IDX_SMALL1] = nh.advertise<std_msgs::Float64>("/rrbot/small1_position_controller/command", 1);
	pub[IDX_SMALL2] = nh.advertise<std_msgs::Float64>("/rrbot/small2_position_controller/command", 1);
	pub[IDX_SMALL3] = nh.advertise<std_msgs::Float64>("/rrbot/small3_position_controller/command", 1);
	pub[IDX_LITTLE1] = nh.advertise<std_msgs::Float64>("/rrbot/little1_position_controller/command", 1);
	pub[IDX_LITTLE2] = nh.advertise<std_msgs::Float64>("/rrbot/little2_position_controller/command", 1);
	pub[IDX_LITTLE3] = nh.advertise<std_msgs::Float64>("/rrbot/little3_position_controller/command", 1);


	Offset[IDX_SHOULDER] = 0;
	Offset[IDX_ELBOW] = 0;
	Offset[IDX_FOREARM] = -1.57;
	Offset[IDX_WRIST1] = 0;
	Offset[IDX_WRIST2] = 0;
	Offset[IDX_THUMB1] = 0;
	Offset[IDX_THUMB2] = 0;
	Offset[IDX_THUMB3] = 0;
	Offset[IDX_INDEX1] = 0;
	Offset[IDX_INDEX2] = 0;
	Offset[IDX_INDEX3] = 0;
	Offset[IDX_MIDDLE1] = 0;
	Offset[IDX_MIDDLE2] = 0;
	Offset[IDX_MIDDLE3] = 0;
	Offset[IDX_SMALL1] = 0;
	Offset[IDX_SMALL2] = 0;
	Offset[IDX_SMALL3] = 0;
	Offset[IDX_LITTLE1] = 0;
	Offset[IDX_LITTLE2] = 0;
	Offset[IDX_LITTLE3] = 0;

	system("rosparam load ~/catkin_ws/src/bhand/config/armControlConfig.yaml");
	
	nh.getParam("Upperbound/Elbow",Ubound[MODE_ELBOW]); 		
	nh.getParam("Upperbound/Forearm",Ubound[MODE_FOREARM]); 	
	nh.getParam("Upperbound/Wrist",Ubound[MODE_WRIST]); 		
	nh.getParam("Upperbound/Hand",Ubound[MODE_HAND]); 		

	nh.getParam("Lowerbound/Elbow",Lbound[MODE_ELBOW]); 	
	nh.getParam("Lowerbound/Forearm",Lbound[MODE_FOREARM]); 	
	nh.getParam("Lowerbound/Wrist",Lbound[MODE_WRIST]); 	
	nh.getParam("Lowerbound/Hand",Lbound[MODE_HAND]); 	


	ROS_INFO("--- PARAMETERS ------------");
	ROS_INFO("Upperbound Elbow : %lf", Ubound[MODE_ELBOW]);
	ROS_INFO("Upperbound Forearm : %lf", Ubound[MODE_FOREARM]);
	ROS_INFO("Upperbound Wrist : %lf", Ubound[MODE_WRIST]);
	ROS_INFO("Upperbound Hand : %lf", Ubound[MODE_HAND]);

	ROS_INFO("Lowerbound Elbow : %lf", Lbound[MODE_ELBOW]);
	ROS_INFO("Lowerbound Forearm : %lf", Lbound[MODE_FOREARM]);
	ROS_INFO("Lowerbound Wrist : %lf", Lbound[MODE_WRIST]);
	ROS_INFO("Lowerbound Hand : %lf", Lbound[MODE_HAND]);
	ROS_INFO("---------------------------");

	emgReady = false;
	CH3_state_pre = 0;
	CH3_state_cur = 0;
	mode = MODE_ELBOW;
	mode_buf = mode;
	posture = POSTURE_GRASP;
	posture_sel_mode = 0;

	while( ros::ok() )
	{
		if( emgReady )
		{
			// 제어 프로토콜
			// CH1 : + 방향(각도) 제어
			// CH2 : - 방향(각도) 제어
			// CH3 : 모드 체인지 ( v : 팔꿈치 , vv : 전완 , vvv : 손목, vvvv : 손 )
			emgReady = false;
			CH3_state_pre = CH3_state_cur;

			if( emgData[CH3] > THRESHOLD )
				CH3_state_cur = 1;
			else
				CH3_state_cur = 0;

			if( CH3_state_pre == 0 && CH3_state_cur == 1 )
			{
				if( CH3_state_time > TIME_THRESHOLD )
					CH3_state_time = 0;
			}
			else if( CH3_state_pre == 1 && CH3_state_cur == 0 )
			{
				if( CH3_state_time > TIME_THRESHOLD )
				{
					CH3_count++;	
					CH3_state_time = 0;
				}
			}
			else if( CH3_state_pre == 0 && CH3_state_cur == 0 )
			{
				CH3_state_time++;

				if( CH3_state_time > 200 )
				{

					if( !posture_sel_mode )
					{
						switch( CH3_count )
						{
							case 0: 						break;
							case 1: mode = MODE_ELBOW; 		break;
							case 2: mode = MODE_FOREARM; 	break;
							case 3: mode = MODE_WRIST; 		break;
							case 4: mode = MODE_HAND; 		break;
							case 5: mode = MODE_HAND;		break;
							default:mode = MODE_HAND; 		break;
						}
					}
					else
					{
						switch( CH3_count )
						{
							case 0: 							break;
							case 1: posture = POSTURE_GRASP;	break;
							case 2: posture = POSTURE_CLICK; 	break;
							default:posture = POSTURE_CLICK; 	break;
						}
					}

					CH3_count = 0;
				}
			}
			else
			{
				CH3_state_time++;

				if( CH3_state_time > 200 )
				{
					if( posture_sel_mode > 0 )
					{
						ROS_INFO("[Posture Selection Mode ] OFF");
						posture_sel_mode = 0;
						mode = mode_buf;
					}
					else
					{
						ROS_INFO("[ Posture Selection Mode ] ON");
						posture_sel_mode = 1;
						mode_buf = mode;
						mode = MODE_NONE;
					}

					CH3_state_time = 0;
				}
			}
			
			// 모터 커맨드 Publish
			
			if( mode == MODE_RESET )
			{
				for(i=0;i<4;i++)
				{
					velocity[i] = 0;
					desired[i] = 0;
				}
			}
			else if( mode == MODE_NONE )
			{
				for(i=0;i<4;i++)
					velocity[i] = 0;
			}
			else
			{
				if( emgData[CH1] > MOVE_THRESHOLD && emgData[CH2] < MOVE_THRESHOLD )
				{
					velocity[mode] = 0.3;
				}
				else if( emgData[CH1] < MOVE_THRESHOLD && emgData[CH2] > MOVE_THRESHOLD )
				{
					velocity[mode] = -0.3;
				}
				else
					velocity[mode] = 0;


				desired[mode] = desired[mode] + velocity[mode]*0.01;

				if( desired[mode] > Ubound[mode] )
					desired[mode] = Ubound[mode];
				else if( desired[mode] < Lbound[mode] )
					desired[mode] = Lbound[mode];
			}
		}

		if( debug_cnt%100 == 0 )
		{
			if( !posture_sel_mode )
			{			
				switch( CH3_count )
				{
					case 0: 
						switch( mode )
						{
							case MODE_ELBOW : ROS_INFO("Elbow  |  Desired : %.3lf  |  Velocity : %.3lf",desired[mode], velocity[mode] ); break;
							case MODE_FOREARM : ROS_INFO("Forearm  |  Desired : %.3lf  |  Velocity : %.3lf",desired[mode], velocity[mode] ); break;
							case MODE_WRIST : ROS_INFO("Wrist  |  Desired : %.3lf  |  Velocity : %.3lf",desired[mode], velocity[mode] ); break;
							case MODE_HAND : ROS_INFO("Hand  |  Desired : %.3lf  |  Velocity : %.3lf",desired[mode], velocity[mode] ); break;
							default : ROS_INFO("Undefined Mode"); break;
						}


						break;

					case 1: ROS_INFO("Count : 1  |  Mode : Elbow "); 	break;
					case 2: ROS_INFO("Count : 2  |  Mode : Forearm"); 	break;
					case 3: ROS_INFO("Count : 3  |  Mode : Wrist"); 	break;
					case 4: ROS_INFO("Count : 4  |  Mode : Hand");		break;
					case 5: ROS_INFO("Count : 5  |  Mode : Reset");		break;
					default:ROS_INFO("Count : -1  |  Mode : None"); 	break;
				}
			}
			else
			{
				switch( CH3_count )
				{
					case 0 : 
						switch( posture )
						{
							case POSTURE_GRASP : ROS_INFO("Count : %d | Posture : Grasping",CH3_count); break;
							case POSTURE_CLICK : ROS_INFO("Count : %d | Posture : Clicking",CH3_count); break;
							default : ROS_INFO("Count : 2 | Posture : Clicking"); break;
						}

						break;

					case 1 : ROS_INFO("Count : 1 | Posture : Grasping");	break;
					case 2 : ROS_INFO("Count : 2 | Posture : Clicking");	break;
					default : ROS_INFO("Count : 2 | Posture : Clicking");	break;
				}
			}
		}

		debug_cnt++;

		if( debug_cnt > 9999 )
			debug_cnt = 0;

		pub_msg[IDX_SHOULDER].data = Offset[IDX_SHOULDER];
		pub_msg[IDX_ELBOW].data = Offset[IDX_ELBOW] + desired[MODE_ELBOW];
		pub_msg[IDX_FOREARM].data = Offset[IDX_FOREARM] + desired[MODE_FOREARM];
		
		pub_msg[IDX_WRIST1].data = Offset[IDX_WRIST1] + desired[MODE_WRIST];
		pub_msg[IDX_WRIST2].data = Offset[IDX_WRIST2] + pub_msg[IDX_WRIST1].data;
		
		if( posture == POSTURE_CLICK )
		{
			pub_msg[IDX_INDEX1].data = Offset[IDX_INDEX1] + desired[MODE_HAND];
			pub_msg[IDX_INDEX2].data = Offset[IDX_INDEX2] + pub_msg[IDX_INDEX1].data*0.7;
			pub_msg[IDX_INDEX3].data = Offset[IDX_INDEX3] + pub_msg[IDX_INDEX1].data*0.7;
		}
		else
		{
			pub_msg[IDX_THUMB1].data = Offset[IDX_THUMB1] + desired[MODE_HAND];
			pub_msg[IDX_THUMB2].data = Offset[IDX_THUMB2] + pub_msg[IDX_THUMB1].data*0.7;
			pub_msg[IDX_THUMB3].data = Offset[IDX_THUMB3] + pub_msg[IDX_THUMB1].data*0.7;

			pub_msg[IDX_INDEX1].data = Offset[IDX_INDEX1] + desired[MODE_HAND];
			pub_msg[IDX_INDEX2].data = Offset[IDX_INDEX2] + pub_msg[IDX_INDEX1].data*0.7;
			pub_msg[IDX_INDEX3].data = Offset[IDX_INDEX3] + pub_msg[IDX_INDEX1].data*0.7;

			pub_msg[IDX_MIDDLE1].data = Offset[IDX_MIDDLE1] + desired[MODE_HAND];
			pub_msg[IDX_MIDDLE2].data = Offset[IDX_MIDDLE2] + pub_msg[IDX_MIDDLE1].data*0.7;
			pub_msg[IDX_MIDDLE3].data = Offset[IDX_MIDDLE3] + pub_msg[IDX_MIDDLE1].data*0.7;

			pub_msg[IDX_SMALL1].data = Offset[IDX_SMALL1] + desired[MODE_HAND];
			pub_msg[IDX_SMALL2].data = Offset[IDX_SMALL2] + pub_msg[IDX_SMALL1].data*0.7;
			pub_msg[IDX_SMALL3].data = Offset[IDX_SMALL3] + pub_msg[IDX_SMALL1].data*0.7;

			pub_msg[IDX_LITTLE1].data = Offset[IDX_LITTLE1] + desired[MODE_HAND];
			pub_msg[IDX_LITTLE2].data = Offset[IDX_LITTLE2] + pub_msg[IDX_LITTLE1].data*0.7;
			pub_msg[IDX_LITTLE3].data = Offset[IDX_LITTLE3] + pub_msg[IDX_LITTLE1].data*0.7;
		}

		for( i = IDX_ELBOW; i <= IDX_LITTLE3 ; i++ )
			pub[i].publish(pub_msg[i]);


#if USE_HW == 1

	#if USE_ARM == 1
			int dir_elbow = v012(velocity[MODE_ELBOW]);
			int dir_forearm = v012(velocity[MODE_FOREARM]);
			int dir_wrist = v012(velocity[MODE_WRIST]) + 1;

			// arm, elbow : current = cur_map*velocity + cur_min
			if( dir_elbow == DIR_PLUS )	//flextion
			{
				arm.Tx( OBJ_WRITE, CANID_ELBOW, 1, current_command, current_max[dir_elbow][0]);
				arm.Tx( OBJ_WRITE, CANID_ELBOW, 2, current_command, current_map[1]*velocity[MODE_ELBOW] + current_min[dir_elbow][1]);
			}
			else if( dir_elbow == DIR_MINUS ) //extension
			{
				arm.Tx( OBJ_WRITE, CANID_ELBOW, 1, current_command, current_map[0]*velocity[MODE_ELBOW] + current_min[dir_elbow][0]);
				arm.Tx( OBJ_WRITE, CANID_ELBOW, 2, current_command, current_max[dir_elbow][1]);
			}
			else
			{
				arm.Tx( OBJ_WRITE, CANID_ELBOW, 2, current_command, 0);
			}

			if( dir_forearm == DIR_PLUS )
			{

			}
			else if( dir_forearm == DIR_MINUS )
			{
			}
			else
			{
			}
			
//			arm.Tx( OBJ_WRITE, CANID_ELBOW, 1, current_command, current_map[0]*velocity[MODE_ELBOW] + current_min[dir_elbow][0]);
//			arm.Tx( OBJ_WRITE, CANID_ELBOW, 2, current_command, current_map[1]*velocity[MODE_ELBOW] + current_min[dir_elbow][1]);
//			arm.Tx( OBJ_WRITE, CANID_FOREARM, 1, current_command, current_map[2]*velocity[MODE_FOREARM] + current_min[dir_forearm][2]);
//			arm.Tx( OBJ_WRITE, CANID_FOREARM, 2, current_command, current_map[3]*velocity[MODE_FOREARM] + current_min[dir_forearm][3]);
			arm.Tx( OBJ_WRITE, CANID_WRIST, 1, wrist_command, dir_wrist);
	#endif

	#if USE_HAND == 1
			//hand : position
			switch( posture )
			{
				case POSTURE_GRASP :
						handdata.s.thumb = desired[MODE_HAND];
						handdata.s.index = desired[MODE_HAND];
						handdata.s.middle = desired[MODE_HAND];
						handdata.s.small = desired[MODE_HAND];
						handdata.s.little = desired[MODE_HAND];
						handdata.s.in_thumb = 0;
						handdata.s.in_fingers = 0;
					break;

				default : 
						handdata.s.index = desired[MODE_HAND];
					break;
			}
	
			hand.writePacket( handdata.bytes, 15 );
	#endif

#endif
		
		loop_rate.sleep();
		ros::spinOnce();
	}


	return 0;
}

void callback_emg(const std_msgs::Int8MultiArray::ConstPtr& arr)
{
	int i = 0;

	for(std::vector<int8_t>::const_iterator it = arr->data.begin(); it != arr->data.end(); ++it )
	{
		emgData[i] = *it;
		i++;
	}
	emgReady = true;
	return;
}

