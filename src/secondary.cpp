
#include "connect.h" 
#include "secondary.h"

extern CommunicationObject commObject;
extern int partyNum;
extern string * addrs;
extern BmrNet ** communicationSenders;
extern BmrNet ** communicationReceivers;
extern void log_print(string str);
#define NANOSECONDS_PER_SEC 1E9

//For time measurements
clock_t tStart;
struct timespec requestStart, requestEnd;
bool alreadyMeasuringTime = false;
int roundComplexitySend = 0;
int roundComplexityRecv = 0;
bool alreadyMeasuringRounds = false;

//For faster modular operations
extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

RSSVectorMyType trainData, testData;
RSSVectorMyType trainLabels, testLabels;
size_t trainDataBatchCounter = 0;
size_t trainLabelsBatchCounter = 0;
size_t testDataBatchCounter = 0;
size_t testLabelsBatchCounter = 0;

size_t INPUT_SIZE;
size_t LAST_LAYER_SIZE;
size_t NUM_LAYERS;

/******************* Main train and test functions *******************/
void parseInputs(int argc, char* argv[])
{	
	if (argc < 6) 
		print_usage(argv[0]);

	partyNum = atoi(argv[1]);

	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = ((i + j) % PRIME_NUMBER);
			subtractModPrime[i][j] = ((PRIME_NUMBER + i - j) % PRIME_NUMBER);
			multiplicationModPrime[i][j] = ((i * j) % PRIME_NUMBER); //How come you give the right answer multiplying in 8-bits??
		}
}

void train(NeuralNetwork* net, NeuralNetConfig* config)
{
	log_print("train");

	for (int i = 0; i < config->numIterations; ++i)
	{
		// cout << "----------------------------------" << endl;  
		// cout << "Iteration " << i << endl;
		readMiniBatch(net, "TRAINING");
		net->forward();
		net->backward();
		// cout << "----------------------------------" << endl;  
	}
}


void test(NeuralNetwork* net)
{
	log_print("test");

	// counter[0]: Correct samples, counter[1]: total samples
	vector<size_t> counter(2,0);
	RSSVectorMyType maxIndex(MINI_BATCH_SIZE);

	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		readMiniBatch(net, "TESTING");
		net->forward();
		net->predict(maxIndex);
		// net->getAccuracy(maxIndex, counter);
	}
}


void loadData(string str)
{
	if (str.compare("MNIST"))
	{
		INPUT_SIZE = 784;
		LAST_LAYER_SIZE = 10;
	}
	else if (str.compare("CIFAR10"))
	{
		INPUT_SIZE = 784;
		LAST_LAYER_SIZE = 10;
	}
	else if (str.compare("ImageNet"))
	{
		INPUT_SIZE = 784;
		LAST_LAYER_SIZE = 10;
	}


	string filename_train_data_next, filename_train_data_prev;
	string filename_test_data_next, filename_test_data_prev;
	string filename_train_labels_next, filename_train_labels_prev;
	string filename_test_labels_next, filename_test_labels_prev;

	if (partyNum < 3)
	{
		filename_train_data_next = "files/train_data_A";
		filename_train_data_prev = "files/train_data_B";
		filename_test_data_next = "files/test_data_A";
		filename_test_data_prev = "files/test_data_B";
		filename_train_labels_next = "files/train_labels_A";
		filename_train_labels_prev = "files/train_labels_B";
		filename_test_labels_next = "files/test_labels_A";
		filename_test_labels_prev = "files/test_labels_B";
	}

	float temp_next = 0, temp_prev = 0;
	ifstream f_next(filename_train_data_next);
	ifstream f_prev(filename_train_data_prev);
	for (int i = 0; i < TRAINING_DATA_SIZE * INPUT_SIZE; ++i)
	{
		f_next >> temp_next; f_prev >> temp_prev;
		trainData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	f_next.close(); f_prev.close();

	ifstream g_next(filename_train_labels_next);
	ifstream g_prev(filename_train_labels_prev);
	for (int i = 0; i < TRAINING_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	{
		g_next >> temp_next; g_prev >> temp_prev;
		trainLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	g_next.close(); g_prev.close();

	ifstream h_next(filename_test_data_next);
	ifstream h_prev(filename_test_data_prev);
	for (int i = 0; i < TRAINING_DATA_SIZE * INPUT_SIZE; ++i)
	{
		h_next >> temp_next; h_prev >> temp_prev;
		testData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	h_next.close(); h_prev.close();

	ifstream k_next(filename_test_labels_next);
	ifstream k_prev(filename_test_labels_prev);
	for (int i = 0; i < TRAINING_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	{
		k_next >> temp_next; k_prev >> temp_prev;
		testLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	}
	k_next.close(); k_prev.close();		

	cout << "Loading data done....." << endl;
}


void readMiniBatch(NeuralNetwork* net, string phase)
{
	size_t s = trainData.size();
	size_t t = trainLabels.size();

	if (phase == "TRAINING")
	{
		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
			net->inputData[i] = trainData[(trainDataBatchCounter + i)%s];

		for (int i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; ++i)
			net->outputData[i] = trainLabels[(trainLabelsBatchCounter + i)%t];

		trainDataBatchCounter += INPUT_SIZE * MINI_BATCH_SIZE;
		trainLabelsBatchCounter += LAST_LAYER_SIZE * MINI_BATCH_SIZE;
	}

	if (trainDataBatchCounter > s)
		trainDataBatchCounter -= s;

	if (trainLabelsBatchCounter > t)
		trainLabelsBatchCounter -= t;



	size_t p = testData.size();
	size_t q = testLabels.size();

	if (phase == "TESTING")
	{
		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
			net->inputData[i] = testData[(testDataBatchCounter + i)%p];

		for (int i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; ++i)
			net->outputData[i] = testLabels[(testLabelsBatchCounter + i)%q];

		testDataBatchCounter += INPUT_SIZE * MINI_BATCH_SIZE;
		testLabelsBatchCounter += LAST_LAYER_SIZE * MINI_BATCH_SIZE;
	}

	if (testDataBatchCounter > p)
		testDataBatchCounter -= p;

	if (testLabelsBatchCounter > q)
		testLabelsBatchCounter -= q;
}

void printNetwork(NeuralNetwork* net)
{
	for (int i = 0; i < net->layers.size(); ++i)
		net->layers[i]->printLayer();
	cout << "----------------------------------------" << endl;  	
}


void selectNetwork(string str, NeuralNetConfig* config, string &ret)
{
	if (str.compare("SecureML") == 0)
	{
		// ret = str;
		// NUM_LAYERS = 3;
		// FCConfig* l0 = new FCConfig(784, MINI_BATCH_SIZE, 128); 
		// FCConfig* l1 = new FCConfig(128, MINI_BATCH_SIZE, 128); 
		// FCConfig* l2 = new FCConfig(128, MINI_BATCH_SIZE, 10); 
		// config->addLayer(l0);
		// config->addLayer(l1);
		// config->addLayer(l2);
	}
	else if (str.compare("Sarda") == 0)
	{
		// ret = str;
		// NUM_LAYERS = 3;
		// ChameleonCNNConfig* l0 = new ChameleonCNNConfig(5,1,5,5,MINI_BATCH_SIZE,28,28,2,2);
		// FCConfig* l1 = new FCConfig(MINI_BATCH_SIZE, 980, 100);
		// FCConfig* l2 = new FCConfig(MINI_BATCH_SIZE, 100, 10);
		// config->addLayer(l0);
		// config->addLayer(l1);
		// config->addLayer(l2);
	}
	else if (str.compare("MiniONN") == 0)
	{
		// ret = str;
		// NUM_LAYERS = 4;
		// CNNConfig* l0 = new CNNConfig(16,1,5,5,MINI_BATCH_SIZE,28,28,2,2);
		// CNNConfig* l1 = new CNNConfig(16,16,5,5,MINI_BATCH_SIZE,12,12,2,2);
		// FCConfig* l2 = new FCConfig(MINI_BATCH_SIZE, 256, 100);
		// FCConfig* l3 = new FCConfig(MINI_BATCH_SIZE, 100, 10);
		// config->addLayer(l0);
		// config->addLayer(l1);
		// config->addLayer(l2);
		// config->addLayer(l3);
	}
	else if (str.compare("LeNet") == 0)
	{
		// ret = str;
		// NUM_LAYERS = 4;
		// CNNConfig* l0 = new CNNConfig(20,1,5,5,MINI_BATCH_SIZE,28,28,2,2);
		// CNNConfig* l1 = new CNNConfig(50,20,5,5,MINI_BATCH_SIZE,12,12,2,2);
		// FCConfig* l2 = new FCConfig(MINI_BATCH_SIZE, 800, 500);
		// FCConfig* l3 = new FCConfig(MINI_BATCH_SIZE, 500, 10);
		// config->addLayer(l0);
		// config->addLayer(l1);
		// config->addLayer(l2);
		// config->addLayer(l3);
	}
	else if (str.compare("AlexNet") == 0)
	{
		// ret = str;
		// NUM_LAYERS = 4;
		// PlainCNNConfig* l0 = new PlainCNNConfig(28,28,1,16,5,1,1,MINI_BATCH_SIZE);
		// PlainCNNConfig* l1 = new PlainCNNConfig(26,26,16,1,2,1,1,MINI_BATCH_SIZE);
		// FCConfig* l2 = new FCConfig(7290, MINI_BATCH_SIZE, 100);
		// FCConfig* l3 = new FCConfig(100, MINI_BATCH_SIZE, 10);
		// config->addLayer(l0);
		// config->addLayer(l1);
		// config->addLayer(l2);
		// config->addLayer(l3);
	}
	else if (str.compare("VGG16") == 0)
	{
		ret = str;
		NUM_LAYERS = 4;
		CNNConfig* l0 = new CNNConfig(28,28,1,16,5,1,1,MINI_BATCH_SIZE);
		CNNConfig* l1 = new CNNConfig(26,26,16,1,2,1,1,MINI_BATCH_SIZE);
		FCConfig* l2 = new FCConfig(7290, MINI_BATCH_SIZE, 100);
		FCConfig* l3 = new FCConfig(100, MINI_BATCH_SIZE, 10);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
	}
	else
		assert(false && "Only SecureML, Sarda, Gazelle, LeNet, AlexNet, and VGG16 Networks supported");
	
}






/********************* COMMUNICATION AND HELPERS *********************/

void start_m()
{
	// cout << endl;
	start_time();
	start_communication();
}

void end_m(string str)
{
	end_time(str);
	pause_communication();
	aggregateCommunication();
	end_communication(str);
}

void start_time()
{
	if (alreadyMeasuringTime)
	{
		cout << "Nested timing measurements" << endl;
		exit(-1);
	}

	tStart = clock();
	clock_gettime(CLOCK_REALTIME, &requestStart);
	alreadyMeasuringTime = true;
}

void end_time(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_time() never called" << endl;
		exit(-1);
	}

	clock_gettime(CLOCK_REALTIME, &requestEnd);
	cout << "------------------------------------" << endl;
	cout << "Wall Clock time for " << str << ": " << diff(requestStart, requestEnd) << " sec\n";
	cout << "CPU time for " << str << ": " << (double)(clock() - tStart)/CLOCKS_PER_SEC << " sec\n";
	cout << "------------------------------------" << endl;	
	alreadyMeasuringTime = false;
}


void start_rounds()
{
	if (alreadyMeasuringRounds)
	{
		cout << "Nested round measurements" << endl;
		exit(-1);
	}

	roundComplexitySend = 0;
	roundComplexityRecv = 0;
	alreadyMeasuringRounds = true;
}

void end_rounds(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_rounds() never called" << endl;
		exit(-1);
	}

	cout << "------------------------------------" << endl;
	cout << "Send Round Complexity of " << str << ": " << roundComplexitySend << endl;
	cout << "Recv Round Complexity of " << str << ": " << roundComplexityRecv << endl;
	cout << "------------------------------------" << endl;	
	alreadyMeasuringRounds = false;
}

void aggregateCommunication()
{
	vector<myType> vec(4, 0), temp(4, 0);
	vec[0] = commObject.getSent();
	vec[1] = commObject.getRecv();
	vec[2] = commObject.getRoundsSent();
	vec[3] = commObject.getRoundsRecv();

	if (partyNum == PARTY_B or partyNum == PARTY_C)
		sendVector<myType>(vec, PARTY_A, 4);

	if (partyNum == PARTY_A)
	{
		receiveVector<myType>(temp, PARTY_B, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
		receiveVector<myType>(temp, PARTY_C, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
	}

	if (partyNum == PARTY_A)
	{
		cout << "------------------------------------" << endl;
		cout << "Total communication: " << (float)vec[0]/1000000 << "MB (sent) and " << (float)vec[1]/1000000 << "MB (recv)\n";
		cout << "Total calls: " << vec[2] << " (sends) and " << vec[3] << " (recvs)" << endl;
		cout << "------------------------------------" << endl;
	}
}


void print_usage (const char * bin) 
{
    cout << "Usage: ./" << bin << " PARTY_NUM IP_ADDR_FILE AES_SEED_INDEP AES_SEED_NEXT AES_SEED_PREV" << endl;
    cout << endl;
    cout << "Required Arguments:\n";
    cout << "PARTY_NUM			Party Identifier (0,1, or 2)\n";
    cout << "IP_ADDR_FILE		\tIP Address file (use makefile for automation)\n";
    cout << "AES_SEED_INDEP		\tAES seed file independent\n";
    cout << "AES_SEED_NEXT		\t \tAES seed file next\n";
    cout << "AES_SEED_PREV		\t \tAES seed file previous\n";
    cout << endl;
    cout << "Report bugs to swagh@princeton.edu" << endl;
    exit(-1);
}

double diff(timespec start, timespec end)
{
    timespec temp;

    if ((end.tv_nsec-start.tv_nsec)<0)
    {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    }
    else 
    {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp.tv_sec + (double)temp.tv_nsec/NANOSECONDS_PER_SEC;
}


void deleteObjects()
{
	//close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}