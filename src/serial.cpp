#include "serial.h"

/*
 * 설명 : 시리얼포트를 연다.
 * 주의 : RTS/CTS 를 제어하지 않는다.
 * 시리얼포트를 열고 이전의 포트설정상태를 저장하지 않았다.
 * */
bool Serial::Open( char *dev_name, int baud, int vtime, int vmin )
{
	struct termios newtio;

	// 시리얼포트를 연다.
	// 읽기/쓰기 모드로 모뎀장치를 연다 : O_RDWR
	// CTRL-C 문자가 오면 프로그램이 종료되지 않도록 controlling tty가 안되도록 한다 : O_NOCTTY
	fd = open( dev_name, O_RDWR | O_NONBLOCK );

	if ( fd < 0 )
	{
		// 화일 열기 실패
		printf( "Device OPEN FAIL %s\n", dev_name );
		return false;
	}

	// 시리얼 포트 환경을 설정한다.
	// BAUDRATE : 전송속도
	// CRTSCTS : 하드웨어 흐름제어
	// CS8 : 8 bit , no parity, 1stop bit
	// CLOCAL : Local connection. 모뎀제어를 하지 않음
	// CREAD : 문자 수신을 가능하게 한다.
	memset(&newtio, 0, sizeof(newtio));
	newtio.c_cflag = CS8 | CLOCAL | CREAD; 

	switch( baud )
	{
		case 460800 : newtio.c_cflag |= B460800; break;
		case 115200 : newtio.c_cflag |= B115200; break;
		case 57600 : newtio.c_cflag |= B57600; break;
		case 38400 : newtio.c_cflag |= B38400; break;
		case 19200 : newtio.c_cflag |= B19200; break;
		case 9600 : newtio.c_cflag |= B9600; break;
		case 4800 : newtio.c_cflag |= B4800; break;
		case 2400 : newtio.c_cflag |= B2400; break;
		default : newtio.c_cflag |= B115200; break;
	}

	// IGNPAR : Parity 에러가 있는 문자 바이트를 무시한다
	// ICRNL : CR문자를 NL문자로 변환처리 한다.( 이처리를 안하면 다른 컴퓨터는 CR문자를 한줄의 종료문자로 인식하지 않을 수 있다.)
	newtio.c_iflag = IGNPAR | ICRNL;

	// Raw Output
	newtio.c_oflag = 0;

	//set input mode (non-canonical, no echo,.....)
	// ICANON : canonical 입력을 가능하게 한다
	// canonical : 한줄 단위로 입력 받음. blocking 없음
	// non-canonical : 정해진 문자수 만큼 받음. 정해진 문자수만큼 들어오기 전까지 blocking
	newtio.c_lflag = 0;

	// 제어 문자 초기화
	// Non-canonical mode에서 ,
	// 		MIN > 0, TIME = 0 : MIN개의 문자를 받을때 까지 무한정 대기
	// 		MIN = 0, TIME > 0 : 0.1*TIME 이 Timeout값이 되며, Timeout전에 한문자라도 들어오면 read는 return
	// 		MIN > 0, TIME > 0 : TIME은 Timeout이 아닌 문자간 타이머로 동작. MIN개의 문자가 들어오거나, 두 문자 사이의 시간이 TIME을 넘으면 return
	// 		MIN = 0, TIME = 0 : read는 즉시 리턴된다. 현재 읽을 수 있는 문자의 개수나 요청한 문자 개수가 반환된다.
	newtio.c_cc[VTIME] = vtime; // 문자간 타이머 또는 Timeout 설정
	newtio.c_cc[VMIN] = vmin; // non-canonical 모드에서 입력받을 문자 수 

	// modem라인을 초기화 하고 포트세팅 마침
	tcflush ( fd, TCIFLUSH );
	tcsetattr( fd, TCSANOW, &newtio );

	return true;
}

/*
   설명 : 시리얼포트를 닫는다.
   */
void Serial::Close()
{
	close( fd );
} 


void Serial::setHeader(int byte_header_, char* header_)
{
	byte_header = byte_header_;

	header = new char[byte_header];
	memcpy(header, header_, byte_header);
}

bool Serial::readPacket(char* buf, int size)
{
	int dwRead;
	char* buf_header = new char[byte_header];

	int err_cnt = 0;

	while(1)
	{
		dwRead = read(fd, &vecRead[0], size);

		if( dwRead <= 0 )
		{
			// ERROR: Time out
			err_cnt++;

			if( err_cnt >= ERROR_THRESHOLD )
			{
				return false;
			}
		}
		else if( dwRead == 1 )
		{
			vecBuf.push_back( vecRead[0] );	
		}
		else
		{
			vecBuf.insert( vecBuf.end(), vecRead.begin(), vecRead.begin() + dwRead ); 
		}

		while( vecBuf.size() >= size + byte_header )
		{
			if( byte_header == 0 )
			{
				memcpy(buf, &vecBuf[0], size);	
				return true;
			}

			memcpy( buf_header, &vecBuf[0], byte_header );

			if( memcmp( buf_header, header, byte_header ) == 0  )
			{
				memcpy( buf, &vecBuf[byte_header], size );
				vecBuf.erase( vecBuf.begin() , vecBuf.begin() + size + byte_header ); 
				return true;
			}
			else
			{
				// ERROR : Packet Missmatch
				vecBuf.erase( vecBuf.begin(), vecBuf.begin() + byte_header );	
				err_cnt++;

				if( err_cnt >= ERROR_THRESHOLD )
				{
					return false;
				}
			}
		}
	}	

	return false;
}

bool Serial::writePacket(char* buf, int size)
{
	char* buf_write = new char[size + byte_header];

	memcpy( buf_write, header, byte_header );
	memcpy( buf_write + byte_header, buf, size );

	return write( fd, buf_write, size + byte_header );
}


int Serial::readNormal(char* buf, int size)
{
	return read( fd, buf, size );
}

int Serial::writeNormal(char *buf, int size)
{
	return write( fd, buf, size );
}
