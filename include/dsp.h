#include<stdlib.h>
#include<math.h>
/*

FIR filter designed with
http://t-filter.appspot.com

sampling frequency: 250 Hz

* 0 Hz - 14 Hz
  gain = 0
  desired attenuation = -20 dB
  actual attenuation = -8.219767184142434 dB

* 15 Hz - 50 Hz
  gain = 1
  desired ripple = 3 dB
  actual ripple = 10.153238647856089 dB

* 51 Hz - 125 Hz
  gain = 0
  desired attenuation = -20 dB
  actual attenuation = -8.219767184142434 dB

*/

#define ORDER 11

float coeff[ORDER] = {
  -0.02752339419570389,
  -0.32743670271702663,
  -0.17359136118627558,
  -0.016479625325254012,
  0.20111475538197918,
  0.2996718860046144,
  0.20111475538197918,
  -0.016479625325254012,
  -0.17359136118627558,
  -0.32743670271702663,
  -0.02752339419570389
};



/*********** fir filter ***************************************/
typedef struct
{
	float data[100];
	float coeff[100];
	int order;
	int top;
	
} FIR_FILTER;

FIR_FILTER fir[16];

void fir_init( FIR_FILTER* f, float *coeff_ , int order )
{
	f->order = order;
	f->top = -1;
	memcpy( f->coeff , coeff_, sizeof(float)*order );
	memset( f->data , 0 , sizeof(float)*100 );
}

float fir_update( FIR_FILTER* f, float data )
{
	float res;
	int i,c;
		
	f->top = (f->top + 1)%f->order;

	f->data[ f->top ] = data;
		
	c = 0;
	res = 0;
	for( i = f->top; i >= 0 ; i-- )
		res += (float)f->data[i] * f->coeff[c++];
	for( i = f->order-1; i > f->top ; i-- )
		res += (float)f->data[i] * f->coeff[c++];
		
	return res; 		
}

void firFilter_init(int num_channel)
{
	int i;
	
	for( i = 0 ; i < num_channel ; i++ )
		fir_init( fir+i, coeff, ORDER );
}

float firFilter_update( int channel, float data )
{
	return fir_update( fir+channel, data );
}


/*********** rms_filter ***************************************/

typedef struct
{
	float data[300];
	int order;
	int top;
	
} rms_filter;

rms_filter rms[16];

void rms_init( rms_filter* f , int order )
{
	f->top = -1;
	f->order = order; 
	memset( f->data , 0 , sizeof(float)*300 );
}

float rms_update( rms_filter* f, float data )
{
	float res;
	int i,c;
		
	f->top = (f->top + 1)%f->order;

	f->data[ f->top ] = data;
		
	res = 0;
	for( i = f->top; i >= 0 ; i-- )
		res += f->data[i] * f->data[i];
	for( i = f->order-1; i > f->top ; i-- )
		res += f->data[i] * f->data[i];
	
	res = sqrt(res) / f->order;

	return res; 		
}

void rmsFilter_init(int window_size)
{
	int i;
	
	for( i = 0 ; i < 16 ; i++ )
		rms_init( rms+i , window_size );
}

float rmsFilter_update( int channel, float data )
{
	return rms_update( rms+channel, data );
}



