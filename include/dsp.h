#include<stdlib.h>

#define ORDER 21 
#define WINDOW_SIZE	32

float coeff[ORDER] = {
	-0.137806477226230378,
		-0.269699749117871368,
		0.107553217049068783,
		0.519116776013301862,
		0.153592279985170888, 
		-0.224451296728918531, 
		-0.024690721956246962, 
		-0.007440408616033959, 
		-0.151650493555151478, 
		-0.016466462220990685, 
		0.032205377141482514, 
		-0.033366235498103830, 
		0.020895681246836620, 
		0.027895853138994828, 
		-0.007052182822281941, 
		0.008181271624559457, 
		0.004518461096413166, 
		-0.003041731074982262, 
		0.002369006607108223, 
		-728.0175199555144440E-6, 
		65.85243383049768800E-6
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

void firFilter_init()
{
	int i;
	
	for( i = 0 ; i < 16 ; i++ )
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
		res += (float)f->data[i];
	for( i = f->order-1; i > f->top ; i-- )
		res += (float)f->data[i];
	
	res = res / f->order;

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



