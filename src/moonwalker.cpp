#include "moonwalker.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

// Variables /////////////////////////////////////////////
uint8_t mType[256] = {  
    INT8,INT16,INT32,FLOAT,FLOAT,FLOAT, INT32,INT32,
    FLOAT,FLOAT,FLOAT,FLOAT,INT32,INT32,INT32,INT32,
    INT32, INT32,INT32,INT16,INT16,INT8,FLOAT,FLOAT,
    FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,INT16,FLOAT,FLOAT,
    FLOAT,INT8,INT8,INT8,INT8,INT8,INT16,INT16,INT16,
    INT8,INT8,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,
    FLOAT,FLOAT,FLOAT,INT16,FLOAT,INT32,0,
};

uint16_t mIdx[256] = {  
  80 ,101,111,112,113,114,102,103,121,122,
  123,124,125,126,131,132,141,142,143,144,
  145,146,151,152,153,154,155,161,162,163,
  164,165,166,167,168,169,173,174,175,176,
  177,171,172,191,192,193,186,187,190,181,
  182,185,90,92,57,0,
};

uint8_t mObj[256];

//////////////////////////////////////////////////////////

bool Moonwalker::init(int channel_)
{
   int i;

   channel = channel_;
   
   // Init mObj[]
   for( i = 0; i < 55 ; i++ )
     mObj[ mIdx[i] ] = i;

   /* Open channels, parameters and go on bus */
   hnd = canOpenChannel(channel, canOPEN_EXCLUSIVE);
   if (hnd < 0) {
	   printf("canOpenChannel %d", channel);
	   check("", (canStatus)hnd);
	   return false;
   }

   stat = canSetBusParams(hnd,canBITRATE_1M, 6, 1, 2, 0, 0);
   check("canSetBusParams", stat);
   if (stat != canOK) {
	  return false; 
   }

   stat = canBusOn(hnd);
   check("canBusOn", stat);
   if (stat != canOK) {
	   return false;
   }	

   return true;
}

// MAKE MESSAGE
MESSAGE Moonwalker::get_message( uint16_t id , uint8_t Command, uint16_t Index, uint8_t SubIndex, float data )
{
   uint8_t type = Command &0x0F;
   MESSAGE res;
  
   memset( res.pkt.bytes, 0, sizeof(uint8_t)*8 );
   
   switch( type )
   {
   case INT8  :  res.pkt.format.data.I8_Value = (int8_t)data;     res.len = 5;    break;
   case INT16 :  res.pkt.format.data.I16_Value = (int16_t)data;   res.len = 6;    break;
   case INT32 :  res.pkt.format.data.I32_Value = (int32_t)data;   res.len = 8;    break;
   case FLOAT :  res.pkt.format.data.F_Value = data;              res.len = 8;    break;
   default :     res.pkt.format.Index = 0;                        res.len = 0;    break; // ERROR
   }
   
//   res.pkt.bytes[0] = Command;
//   res.pkt.bytes[1] = Index&0xff;
//   res.pkt.bytes[2] = (Index<<8)&0xff;
//   res.pkt.bytes[3] = SubIndex;
//   
   res.pkt.format.Command = Command;
   res.pkt.format.Index = Index;
   res.pkt.format.SubIndex = SubIndex;
   res.id = id;

   return res;
}

void Moonwalker::Tx(uint8_t cmd_rw, uint16_t id, uint8_t motor_num, uint8_t obj, float value)
{
	MESSAGE msg;
    uint8_t cmd = cmd_rw | mType[obj];
   	 
    msg = get_message( id, cmd, mIdx[obj], motor_num, value );

//	printf("command : %x\n",msg.pkt.format.Command);
//	printf("Index : %d\n",msg.pkt.format.Index);
//	printf("SubIndex : %d\n",msg.pkt.format.SubIndex);
//	printf("data_I8 : %d\n",msg.pkt.format.data.I8_Value);
//	printf("data_I16 : %d\n",msg.pkt.format.data.I16_Value);
//	printf("data_I32 : %d\n",msg.pkt.format.data.I32_Value);
//	printf("data_F : %f\n\n",msg.pkt.format.data.F_Value);

	stat = canWrite(hnd, msg.id , msg.pkt.bytes , msg.len, canMSG_STD);
	check("canWrite",stat);
}


float Moonwalker::read( uint16_t id, uint8_t motor_num, uint8_t obj )
{
    float res = 0;
    
    switch( obj )
    {
    case num_motors:    res = (float)drv[ id ].num_motors;              break;
    case status:        res = (float)drv[ id ].status[ motor_num-1 ];   break;
    case fault:         res = (float)drv[ id ].fault[ motor_num-1 ];    break;
    case position:      res = (float)drv[ id ].pos[ motor_num-1 ];      break;
    case direction:     res = (float)drv[ id ].dir[ motor_num-1 ];      break;
    }
    
    return res;
}

void Moonwalker::Copy2drv( uint8_t drv_id, uint8_t sub_idx, uint8_t obj, uint8_t obj_type, uint8_t *data )
{
    uint8_t byte_size = 0;
    
    switch( obj_type )
    {
    case INT8: byte_size = 1;   break;
    case INT16: byte_size = 2;  break;
    case INT32:
    case FLOAT: byte_size = 4;  break;
    default : break;
    }
    
    switch( obj )
    {
    case num_motors:    memcpy( &drv[ drv_id ].num_motors , data , byte_size );          break; 
    case status:        memcpy( &drv[ drv_id ].status[ sub_idx-1 ] , data , byte_size );  break;
    case fault:         memcpy( &drv[ drv_id ].fault[ sub_idx-1 ] , data , byte_size );   break;
    case position:      memcpy( &drv[ drv_id ].pos[ sub_idx-1 ] , data ,  byte_size );    break;
    case direction:     memcpy( &drv[ drv_id ].dir[ sub_idx-1 ] , data , byte_size );     break;
    }
}

void Moonwalker::Rx()
{
    uint8_t accessCode, objType, motorNum, objName;
    uint8_t *addValue;
 
	long id;
	unsigned char msgin[4]={0,};
	unsigned int dlc;
	unsigned int flag;
	unsigned long time;
	PACKET *temp = (PACKET*)&msgin[0];
	int idx;

    // get packet from CAN2
 	stat = canRead(hnd, &id, &msgin, &dlc, &flag, &time); 

	// disassembly packet
    accessCode = temp->format.Command & 0xF0;
    objType = temp->format.Command & 0x0F;
    motorNum = temp->format.SubIndex;
    objName = mObj[ temp->format.Index ];
    addValue = temp->format.data.bytes;

    // data process
    switch( accessCode )
    {
    case OBJ_READ_R:
      Copy2drv( id , motorNum, objName, objType, addValue );

//	  printf("Rx Command : %x\n", temp->format.Command);
//	  printf("Rx Index : %d\n\n", temp->format.Index);

	  readFlag = true;
      break;
    case OBJ_WRITE_R:
//	  printf("Rx Command : %x\n", temp->format.Command);
//	  printf("Rx Index : %d\n", temp->format.Index);
//	  printf("Rx I8_Value : %d\n", temp->format.data.I8_Value);
//	  printf("Rx I16_Value : %d\n", temp->format.data.I16_Value);
//	  printf("Rx I32_Value : %d\n", temp->format.data.I32_Value);
//	  printf("Rx F_Value : %.2f\n\n", temp->format.data.F_Value);

      break;
    case OBJ_RW_ERR:
//	  printf("Rx Command : %x\n", temp->format.Command);
//	  printf("Rx Index : %d\n\n", temp->format.Index);
      break;
    default:
      break;
    }
}

void Moonwalker::check(char* id, canStatus stat)
{
  if (stat != canOK) {
    char buf[50];
    buf[0] = '\0';
    canGetErrorText(stat, buf, sizeof(buf));
    printf("%s: failed, stat=%d (%d)\n", id, (int)stat, buf);
  }
}
