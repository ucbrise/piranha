
#include "connect.h"
#include "util.cuh"

#include <mutex> 
#include <thread>
#include <vector>

#define STRING_BUFFER_SIZE 256
//extern void error(std::string str);

//this player number
extern int partyNum;

//communication
std::string *addrs;
BmrNet **communicationSenders;
BmrNet **communicationReceivers;

//Communication measurements object
extern CommunicationObject commObject;

extern nlohmann::json piranha_config;

//setting up communication
void initCommunication(std::string addr, int port, int player, int mode)
{
	char temp[25];
	strcpy(temp, addr.c_str());
	if (mode == 0)
	{
		communicationSenders[player] = new BmrNet(temp, port);
		communicationSenders[player]->connectNow();
	}
	else
	{
		communicationReceivers[player] = new BmrNet(port);
		communicationReceivers[player]->listenNow();
	}
}

void initializeCommunication(int* ports, int num_parties) {
	int i;
	communicationSenders = new BmrNet*[num_parties];
	communicationReceivers = new BmrNet*[num_parties];
    std::thread *threads = new std::thread[num_parties * 2];
	for (i = 0; i < num_parties; i++)
	{
		if (i != partyNum)
		{
			threads[i * 2 + 1] = std::thread(initCommunication, addrs[i], ports[i * 2 + 1], i, 0);
			threads[i * 2] = std::thread(initCommunication, "127.0.0.1", ports[i * 2], i, 1);
		}
	}
	for (int i = 0; i < 2 * num_parties; i++)
	{
		if (i != 2 * partyNum && i != (2 * partyNum + 1))
			threads[i].join();//wait for all threads to finish
	}

	delete[] threads;
}

void initializeCommunicationSerial(int *ports, int num_parties) { //Use this for many parties
	communicationSenders = new BmrNet*[num_parties];
	communicationReceivers = new BmrNet*[num_parties];
	for (int i = 0; i < num_parties; i++)
	{
		if (i<partyNum)
		{
		  initCommunication( addrs[i], ports[i * 2 + 1], i, 0);
		  initCommunication("127.0.0.1", ports[i * 2], i, 1);
		}
		else if (i>partyNum)
		{
		  initCommunication("127.0.0.1", ports[i * 2], i, 1);
		  initCommunication( addrs[i], ports[i * 2 + 1], i, 0);
		}
	}
}

void initializeCommunication(char* filename, int party, int num_parties) {

	FILE *f = fopen(filename, "r");
	partyNum = party;
	char buff[STRING_BUFFER_SIZE];
	char ip[STRING_BUFFER_SIZE];
	
	addrs = new std::string[num_parties];
	int *ports = new int[num_parties * 2];

	for (int i = 0; i < num_parties; i++) {
		if (fgets(buff, STRING_BUFFER_SIZE, f)) {
            sscanf(buff, "%s\n", ip);
            addrs[i] = std::string(ip);
            ports[2 * i] = 32000 + i*num_parties + partyNum;
            ports[2 * i + 1] = 32000 + partyNum*num_parties + i;
        } else {
            std::cout << "initializeCommunication: error: IP file empty" << std::endl;
        }
	}

	fclose(f);
	initializeCommunicationSerial(ports, num_parties);

	delete[] ports;
}

void initializeCommunication(std::vector<std::string> &ips, int party, int num_parties) {

	addrs = new std::string[num_parties];
	int *ports = new int[num_parties * 2];

    for (int i = 0; i < num_parties; i++) {
        addrs[i] = ips[i];
        ports[2 * i] = 32000 + i * num_parties + party;
        ports[2 * i + 1] = 32000 + party * num_parties + i;
    }

	initializeCommunicationSerial(ports, num_parties);

	delete[] ports;
}

//synchronization functions
void sendByte(int player, char* toSend, int length, int conn)
{
	communicationSenders[player]->sendMsg(toSend, length, conn);
	// totalBytesSent += 1;
}

void receiveByte(int player, int length, int conn)
{
	char *sync = new char[length+1];
	communicationReceivers[player]->receiveMsg(sync, length, conn);
	delete[] sync;
	// totalBytesReceived += 1;
}

void synchronize(int length, int num_parties) {
	char* toSend = new char[length+1];
	memset(toSend, '0', length+1);
    std::vector<std::thread *> threads;
	for (int i = 0; i < num_parties; i++)
	{
		if (i == partyNum) continue;
		for (int conn = 0; conn < NUMCONNECTIONS; conn++)
		{
			threads.push_back(new std::thread(sendByte, i, toSend, length, conn));
			threads.push_back(new std::thread(receiveByte, i, length, conn));
		}
	}
	for (auto it = threads.begin(); it != threads.end(); it++)
	{
		(*it)->join();
		delete *it;
	}
	threads.clear();
	delete[] toSend;
}

void start_communication()
{
	if (commObject.getMeasurement())
		error("Nested communication measurements");

	commObject.reset();
	commObject.setMeasurement(true);
}

void pause_communication()
{
	if (!commObject.getMeasurement())
		error("Communication never started to pause");

	commObject.setMeasurement(false);
}

void resume_communication()
{
	if (commObject.getMeasurement())
		error("Communication is not paused");

	commObject.setMeasurement(true);
}

void end_communication(std::string str)
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Communication, " << str << ", P" << partyNum << ": " 
		 << (float)commObject.getSent()/1000000 << "MB (sent) " 
		 << (float)commObject.getRecv()/1000000 << "MB (recv)" << std::endl;
    std::cout << "Rounds, " << str << ", P" << partyNum << ": " 
		 << commObject.getRoundsSent() << "(sends) " 
		 << commObject.getRoundsRecv() << "(recvs)" << std::endl; 
    std::cout << "----------------------------------------------" << std::endl;	
	commObject.reset();
}
