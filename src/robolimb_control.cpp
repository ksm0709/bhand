#include <ros/ros.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/Int8.h>
#include "robolimb/RobolimbState.h"
#include <stdlib.h>
#include <vector>

#define NUM_POSTURE 10 + 1
enum POSTURE
{
   rest = 1, pinch, fist, hook, pointing, one, two, three, four, five
};

#define THUMB		0
#define INDEX		1
#define	MIDDLE		2
#define RING		3
#define LITTLE		4
#define ROTATOR		5

#define NORMAL_GRIP                             0
#define STANDARD_PRECISION_PINCH_CLOSED         1
#define STANDARD_TRIPOD_CLOSED                  2
#define THUMB_PARK                              3
#define LATERAL_GRIP                            5
#define INDEX_POINT                             6
#define STANDARD_PRECISION_PINCH_OPENED         7
#define THUMB_PRECISION_PINCH_CLOSED            9
#define THUMB_PRECISION_PINCH_OPENED            10
#define THUMB_TRIPOD_CLOSED                     11
#define STANDARD_TRIPOD_OPENED                  13
#define THUMB_TRIPOD_OPENED                     14
#define COVER_SETTING                           18

#define STATE_STOP				0
#define STATE_CLOSING			1
#define STATE_OPENING			2
#define STATE_STALLED_CLOSED	3
#define STATE_STALLED_OPEN		4
#define STATE_THUMB_ROT_NOT_EDGE	0
#define STATE_THUMB_ROT_EDGE		1
#define STATE_ANY               5

#define MODE_QUICK	0
#define MODE_PWM	1

#define STOP 0
#define OPEN -150
#define OPEN_FAST -255
#define CLOSE 150
#define CLOSE_FAST 255

typedef struct
{
    uint8_t thumb, index, middle, little, small, rot;
} des_state;

ros::Subscriber sub_state;
ros::Subscriber sub_posture; 
ros::Publisher pub_cmd; 

robolimb::RobolimbState hand;
std_msgs::Int16MultiArray cmd;
des_state state;
bool subFlag = false;
int posture; 
int count[NUM_POSTURE];
int posture_num=0;

void posture_callback(const std_msgs::Int8::ConstPtr& msg);
void state_callback(const robolimb::RobolimbState::ConstPtr& msg);

int max_index(int* arr, int size);
void publish(int16_t mode, int16_t p1, int16_t p2, int16_t p3, int16_t p4, int16_t p5, int16_t p6);
void state_wait(double timelimit);
void state_init();

uint8_t state_stalled(short dir);
void move(short dirt,short diri, short dirm, short dirl, short dirs, short dirr, double t);
void extension();
void grip();
void rot(short dir, double t);
void index(short dir, double t);
void thumb(short dir, double t);

int main(int argc, char** argv)
{
	ros::init(argc,argv,"robolimb_controller");
	ros::NodeHandle nh;

	sub_state = nh.subscribe("robolimb_state",10,state_callback);
	sub_posture = nh.subscribe("posture",10,posture_callback);
	pub_cmd = nh.advertise<std_msgs::Int16MultiArray>("robolimb_cmd",10);

	ros::Rate loop_rate(1);

	if( argc > 2 )
	{
        ROS_INFO("Command publishing!");
        while( ros::ok() )
        {
            publish( atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
            loop_rate.sleep();
            ros::spinOnce();
        }
        return 0;
	}
    else if( argc == 2)
        {
            posture_num = 999;
            count[ atoi(argv[1]) ] = 999;
            subFlag = true;
        }

    int pre_idx=999;
	ROS_INFO("Robolibm control start!");
    state_init();
    state_wait(0.5);
    publish( MODE_QUICK, 0, 0, 0, 0, 0, 0 );
    state_wait(0.5);

	while( ros::ok() )
	{
		if( subFlag )
		{
			if( posture_num < 10 )
			{
				count[ posture ]++;		 	
				posture_num++;
			}	
			else
			{
				int idx = max_index( count , NUM_POSTURE );

                if( pre_idx == idx ) idx = 0;
                else                 pre_idx = idx;

				ROS_INFO("Execute Posture %d", idx );
				switch( idx	)
				{
                case rest: 
                    extension(); 
                    thumb(CLOSE_FAST,1);
                break;
                case pinch: 
                    extension();
                    rot(CLOSE_FAST,10);
                    rot(OPEN_FAST, 0.12);
                    thumb(CLOSE,1.1);
                    index(CLOSE,10);
                break;
                case fist: 
                    extension();
                    rot(CLOSE_FAST,10);
                    grip();
                break;
                case hook: 
                    extension();
                    move(STOP,CLOSE_FAST,CLOSE_FAST,CLOSE_FAST,CLOSE_FAST,STOP,10); 
                break;
                case pointing: 
                    extension();
                    move(STOP,STOP,CLOSE_FAST,CLOSE_FAST,CLOSE_FAST,STOP,10); 
                break;
                case one: 
                    extension();
                    rot(CLOSE_FAST,0.5);
                    move(CLOSE_FAST,STOP,CLOSE_FAST,CLOSE_FAST,CLOSE_FAST,STOP,10); 
                break;
                case two: 
                    extension();
                    rot(CLOSE_FAST,0.5);
                    move(CLOSE_FAST,STOP,STOP,CLOSE_FAST,CLOSE_FAST,STOP,10); 
                break;
                case three: 
                    extension();
                    rot(CLOSE_FAST,0.5);
                    move(CLOSE_FAST,STOP,STOP,STOP,CLOSE_FAST,STOP,10); 
                break;
                case four: 
                    extension();
                    rot(CLOSE_FAST,0.5);
                    move(CLOSE_FAST,STOP,STOP,STOP,STOP,STOP,10); 
                break;
                case five: 
                    extension();
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

void state_callback(const robolimb::RobolimbState::ConstPtr& msg)
{
    hand = *msg;
}

void posture_callback(const std_msgs::Int8 ::ConstPtr& msg)
{
	posture = msg->data;	
	subFlag = true;
}

void state_wait(double timelimit)
{
    int i;
    bool flag, flag_all_any;
    double ts;
    uint8_t* desired_pt = (uint8_t*)&state;

    ros::Rate loop_rate(1000);
    ROS_INFO("WAIT %.3lf",timelimit);

    flag_all_any = true;
    for(i=0;i<6;i++)
        if( desired_pt[i] != STATE_ANY )
        {
           flag_all_any = false;
           break; 
        }

    ts = ros::Time::now().toSec();
    while( ros::ok() )
    {
        ros::spinOnce(); 
        loop_rate.sleep();

        if( ros::Time::now().toSec()-ts >= timelimit ) break;

        if( !flag_all_any )
        {
            flag = true;
            for(i=0;i<6;i++)
                if( desired_pt[i] != hand.state[i] && desired_pt[i] != STATE_ANY )
                    flag = false;

            if( flag ) break;
        }

    }

    
}

void state_init()
{
    state.thumb = STATE_ANY;
    state.middle = STATE_ANY;
    state.index = STATE_ANY;
    state.small = STATE_ANY;
    state.little = STATE_ANY;
    state.rot = STATE_ANY;
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

void publish( int16_t mode, int16_t p1, int16_t p2, int16_t p3, int16_t p4, int16_t p5, int16_t p6)
{
	cmd.data.clear();

	cmd.data.push_back( mode );
	cmd.data.push_back( p1 );
	cmd.data.push_back( p2 );
	cmd.data.push_back( p3 );
	cmd.data.push_back( p4 );
	cmd.data.push_back( p5 );
	cmd.data.push_back( p6 );

	pub_cmd.publish( cmd );
}

uint8_t state_stalled(short dir)
{
    if( dir > 0 )
        return STATE_STALLED_CLOSED;
    else if( dir < 0 )
        return STATE_STALLED_OPEN;
    else
        return STATE_ANY;
}

void move(short dirt,short diri, short dirm, short dirl, short dirs, short dirr, double t)
{
    ROS_INFO("MOVE : %d %d %d %d %d %d",dirt,diri,dirm,dirl,dirs,dirr);
    state_init();
    publish( MODE_PWM, dirt, diri, dirm, dirl, dirs, dirr );
    state.thumb = state_stalled(dirt);
    state.index = state_stalled(diri);
    state.middle = state_stalled(dirm);
    state.little = state_stalled(dirl);
    state.small = state_stalled(dirs);
    state.rot = state_stalled(dirr);
    state_wait(t);
    publish(MODE_PWM, STOP, STOP, STOP, STOP, STOP, STOP);
    state_wait(0.01);
}

void extension()
{
    move(OPEN_FAST, OPEN_FAST, OPEN_FAST, OPEN_FAST, OPEN_FAST, OPEN_FAST,10 );
}
void grip()
{
    move(CLOSE, CLOSE_FAST, CLOSE_FAST, CLOSE_FAST, CLOSE_FAST, 0, 10);
}
void rot(short dir, double t)
{
    move(0,0,0,0,0,dir,t);
}
void index(short dir, double t)
{
    move(0,dir,0,0,0,0,t);
}
void thumb(short dir, double t)
{
    move(dir,0,0,0,0,0,t);
}