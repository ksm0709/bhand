/*
 * 설 명 : 시리얼 포트를 사용하여 데이타를 전송하는 예제이다.
 * */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/signal.h>
#include <sys/ioctl.h>
#include <sys/poll.h>

#include <termios.h>
#include <vector>

#define ERROR_THRESHOLD		200

class Serial
{
public:
	Serial()
	{
		vecRead.assign(1000, 0);
		byte_header = 0;
		header = NULL; 
	}
	
	Serial(int byte_header_, char* header_)
	{
		vecRead.assign(1000, 0);
		byte_header = byte_header_;
		
		header = new char[byte_header];
		memcpy(header, header_, byte_header);
	}

	~Serial()
	{
		Close();
	}

	bool Open( char *dev_name, int baud, int vtime, int vmin );
	void Close();

	void setHeader(int byte_header_, char* header_);
	
	bool readPacket(char* buf, int size);
	bool writePacket(char* buf, int size);
	int readNormal(char *buf, int size);
	int writeNormal(char *buf, int size);

private:
	int fd;	

	int byte_header;
	char* header;	

	std::vector<char> vecBuf;
	std::vector<char> vecRead;
};
