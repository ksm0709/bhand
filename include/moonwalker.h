#ifndef _MWK_
#define _MWK_
#include <stdint.h>
#include <canlib.h>

typedef union{
  uint8_t bytes[4];
  
  int8_t I8_Value;
  int16_t I16_Value;
  int32_t I32_Value;
  float F_Value;
} PKT_DATA;

typedef union
{
    uint8_t bytes[8];
    
    struct{
      uint32_t Command  :8;
      uint32_t Index   :16;
      uint32_t SubIndex :8;
      PKT_DATA data;    // 32bit
    }format;
} PACKET;


typedef struct
{
  PACKET pkt;
  uint8_t len;
  uint16_t id;
} MESSAGE;

typedef struct
{
  int8_t  num_motors;
  int32_t status[2];
  int32_t fault[2];
  int32_t pos[2];
  int8_t dir[2];
} DRIVER;



/****************** DRIVER DEFS *******************************************/

// ACCESS CODE
#define OBJ_WRITE      0x10
#define OBJ_WRITE_R    0x20
#define OBJ_READ       0x30
#define OBJ_READ_R     0x40
#define OBJ_RW_ERR     0x80

// RW ERROR CODE
#define NO_ERROR                0x00
#define UNDEFINED_INDEX         0x01
#define PACKET_FORMAT_ERROR     0x02
#define ACCESS_ERROR            0x03

// OBJECT TYPE
#define INT8      0x00
#define INT16     0x04
#define INT32     0x08
#define FLOAT     0x0C


/******************* PRE DEFINED PACKET COMPONENTS  **********************/

// Object Long Names /////////////////////////////////////////////////////
enum{
    num_motors = 0x00,
    command,
    position_command,
    velocity_command,
    current_command,
    voltage_command,
    status,
    fault,
    temperature,
    voltage,
    current,
    velocity,
    position,
    hall_count,
    ai_potentiometer,
    ai_tachometer,
    min_position,
    max_position,
    home_position,
    encoder_ppr,
    num_pole_pairs,
    use_soft_limit,
    max_current,
    max_voltage,
    max_velocity,
    acceleration,
    deceleration,
    overheat_limit,
    overcurrent_limit,
    overcurrent_delay,
    peakcurrent_ratio,
    overvoltage_limit,
    undervoltage_limit,
    stall_detection,
    vel_error_detection,
    pos_error_detection,
    startup_power_on,
    direction,
    brake_on_delay,
    high_voltage,
    high_temperature,
    feedback_sensor,
    profile_mode,
    pc_kp,
    pc_ki,
    pc_kd,
    vc_kp,
    vc_ki,
    vc_ks,
    cc_kp,
    cc_ki,
    cc_kff,
	posture_command,
	hand_command,
	wrist_command
};

// Index & Data Type //////////////////////////////////////////////////////

extern uint8_t mType[256];      // mType[ object_name ] : data type
extern uint16_t mIdx[256];       // mIdx[ object_name ] : name -> index
extern uint8_t mObj[256];       // mObj[ object_index ] : index -> name

// Sub Index ( Motor Num ) ////////////////////////////////////////////////

#define MOTOR1   1
#define MOTOR2   2

// Data , Code ////////////////////////////////////////////////////////////

#define DRIVER_CMD_MOTOR_OFF            0
#define DRIVER_CMD_MOTOR_ON             1
#define DRIVER_CMD_CLEAR_FAULT_FLAG     2
#define DIRVER_CMD_DECLERATION_STOP     6
#define DIRVER_CMD_QUICK_STOP           7

#define DRIVER_STAT_MOTOR_ON                    0x0001
#define DRIVER_STAT_MOTOR_MOVING                0x0002
#define DRIVER_STAT_FAULT_DETECT                0x0004
#define DRIVER_STAT_EMERG_STOP                  0x0008
#define DRIVER_STAT_POS_CTR                     0x0010
#define DRIVER_STAT_VEL_CTR                     0x0020
#define DRIVER_STAT_CUR_CTR                     0x0040
#define DRIVER_STAT_AI_SAFETY_CHECK_FAIL        0x0100
#define DRIVER_STAT_PI_SAFETY_CHECK_FAIL        0x0200
#define DRIVER_STAT_AI_MINMAX_CHECK_FAIL        0x0400
#define DRIVER_STAT_PI_MINMAX_CHECK_FAIL        0x0800
#define DRIVER_STAT_DI_STOP_REQ                 0x1000
#define DRIVER_STAT_EXEED_LIMIT                 0x2000
#define DRIVER_STAT_SERIAL_CMD                  0x4000
#define DRIVER_STAT_AIPI_CMD                    0x8000
#define DRIVER_STAT_SCRIPT_RUN                  0x10000

#define FAULT_STAT_OVER_CUR              0x0001
#define FAULT_STAT_OVER_VOLT             0x0002
#define FAULT_STAT_UNDER_VOLT            0x0004
#define FAULT_STAT_OVER_HEAT             0x0008
#define FAULT_STAT_SHORT_CIRCUIT         0x0010
#define FAULT_STAT_STALL_DETECTION       0x0020
#define FAULT_STAT_VEL_ERR_DETECTION    0x0040
#define FAULT_STAT_POS_ERR_DETECTION    0x0080

#define USE_SOFT_LIMIT          1
#define OFF_SOFT_LIMIT      0

#define STALL_OFF       0
#define STALL_100MS     1
#define STALL_200MS     2
#define STALL_400MS     3
#define STALL_700MS     4
#define STALL_1S        5

#define VEL_ERR_OFF     0
#define VEL_ERR_100MS   1
#define VEL_ERR_200MS   2
#define VEL_ERR_400MS   3
#define VEL_ERR_700MS   4
#define VEL_ERR_1S      5

#define POS_ERR_OFF     0
#define POS_ERR_100MS   1
#define POS_ERR_200MS   2
#define POS_ERR_400MS   3
#define POS_ERR_700MS   4
#define POS_ERR_1S      5

#define STARTUP_POWER_ON_OFF    0
#define STARTUP_POWER_ON_ON     1

#define DIRECTION_FW    0
#define DIRECTION_BW    1

#define SENSOR_NONE     0
#define SENSOR_ENCODER  1
#define SENSOR_HALL     2
#define SENSOR_POTEN    3
#define SENSOR_TACHO    4

#define PROFILE_OFF     0
#define PROFILE_ON      1

/**********************************************************************/

// Driver Defs ////////////////////////////////////////////////////////

#define MWK1_ID         1
#define MWK2_ID         2
#define MWK3_ID         3

///////////////////////////////////////////////////////////////////////



/*********************************************************************/
#define DRV_NUM_MAX     10

class Moonwalker
{
public:
	Moonwalker(){ readFlag = false; };
	Moonwalker(int channel_){ init(channel_); readFlag = false; }

private:
	canHandle hnd;
	canStatus stat;
	int channel;

	DRIVER drv[DRV_NUM_MAX];
	uint8_t drv_num;

	void check(char* id, canStatus stat);
	void Copy2drv( uint8_t drv_id, uint8_t sub_idx, uint8_t obj, uint8_t obj_type, uint8_t *data );
	MESSAGE get_message( uint16_t id, uint8_t Command, uint16_t Index, uint8_t SubIndex, float data );

public:
	bool readFlag;

	bool init(int channel);
	float read( uint16_t id, uint8_t motor_num, uint8_t obj ); // read values
	void Rx();	// receive msg & set values
	void Tx(uint8_t cmd_rw, uint16_t id, uint8_t motor_num, uint8_t obj, float value);
};


#endif
