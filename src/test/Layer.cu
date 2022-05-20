
#include "unitTests.h"

#include "../util/model.h"
#include "../util/functors.h"

#ifndef LAYER_TEST_SHARE
#define LAYER_TEST_SHARE RSS
#endif

template<typename T>
struct LayerTest : public testing::Test {
    using ParamType = T;
};

TYPED_TEST_CASE(LayerTest, uint64_t);

TYPED_TEST(LayerTest, FCForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputDim = 4;
    int batchSize = 4;
    int outputDim = 3;

    LAYER_TEST_SHARE<T> input {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    FCConfig lconfig(inputDim, batchSize, outputDim);
    FCLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    layer.forward(input);

    LAYER_TEST_SHARE<T> temp(batchSize * outputDim);
    temp += *layer.getWeights();

    uint64_t bias = (uint64_t) (0.01 * (1 << FLOAT_PRECISION));
    temp += bias;

    std::vector<double> expected(batchSize * outputDim);
    copyToHost(temp, expected);

    assertShare(*layer.getActivation(), expected, true);
}

TYPED_TEST(LayerTest, FCForwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputDim = 16;
    int batchSize = 128;
    int outputDim = 10;

    FCConfig lconfig(inputDim, batchSize, outputDim);
    FCLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    // load weights and biases
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_weight", *layer.getWeights());
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_bias", *layer.getBiases());

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * inputDim);
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_input", input);

    layer.forward(input);

    // load expected output and compare with calculated activations
    LAYER_TEST_SHARE<T> expected(batchSize * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_output", expected);

    //printShareTensor(*layer.getActivation(), "activations", 128, 1, 1, 10);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected);
    assertShare(*layer.getActivation(), host_expected);
}

TYPED_TEST(LayerTest, FCForwardSecureML) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputDim = 784;
    int batchSize = 128;
    int outputDim = 128;

    FCConfig lconfig(inputDim, batchSize, outputDim);
    FCLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    // load weights and biases
    loadShareFromFile(std::string(TEST_PATH) + "secureml/weight0", *layer.getWeights());
    loadShareFromFile(std::string(TEST_PATH) + "secureml/bias0", *layer.getBiases());

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * inputDim);
    loadShareFromFile(std::string(TEST_PATH) + "secureml/input", input);

    layer.forward(input);

    // load expected output and compare with calculated activations
    LAYER_TEST_SHARE<T> expected(batchSize * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "secureml/outputlayer0", expected);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected);
    assertShare(*layer.getActivation(), host_expected);
}

TYPED_TEST(LayerTest, FCBackwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    log_learning_rate = 6;

    int inputDim = 16;
    int batchSize = 128;
    int outputDim = 10;

    FCConfig lconfig(inputDim, batchSize, outputDim);
    FCLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    // load weights and biases
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_weight", *layer.getWeights());
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_bias", *layer.getBiases());

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * inputDim);
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_input", input);

    // load gradient in
    LAYER_TEST_SHARE<T> gradin(batchSize * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_gradin", gradin);

    //printShare(*layer.getWeights(), "weights?");

    layer.backward(gradin, input);

    // check matching gradout
    LAYER_TEST_SHARE<T> gradout(batchSize * inputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_gradout", gradout);

    std::vector<double> expected_gradout(gradout.size());
    copyToHost(gradout, expected_gradout);
    //printf("asserting gradout\n");
    assertShare(*layer.getDelta(), expected_gradout);

    // check weight update
    LAYER_TEST_SHARE<T> new_weights(inputDim * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_newweight", new_weights);

    std::vector<double> expected_new_weights(new_weights.size());
    copyToHost(new_weights, expected_new_weights);
    //printf("asserting new weights\n");
    //printShare(*layer.getWeights(), "actual weights");
    //printShare(new_weights, "expected weights");
    assertShare(*layer.getWeights(), expected_new_weights);

    // check bias update
    LAYER_TEST_SHARE<T> new_biases(outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_newbias", new_biases);

    LAYER_TEST_SHARE<T> grad_biases(outputDim);
    loadShareFromFile(std::string(TEST_PATH) + "test_fclayer_gradbias", grad_biases);

    std::vector<double> expected_new_biases(new_biases.size());
    copyToHost(new_biases, expected_new_biases);
    //printf("asserting new biases\n");
    assertShare(*layer.getBiases(), expected_new_biases);
}

TYPED_TEST(LayerTest, ReLUForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        -2, -3, 4, 3, 3.5, 1, -1.5, -1, 2, 1 << 12, -17, (1 << 18) - 1
    };

    ReLUConfig lconfig(
        input.size(),
        1 // "batch" size 
    );
    ReLULayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(input);

    std::vector<double> expected = {
        0, 0, 4, 3, 3.5, 1, 0, 0, 2, 1 << 12, 0, (1 << 18) - 1
    };
    assertShare(*layer.getActivation(), expected);
}

TYPED_TEST(LayerTest, ReLUForwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputDim = 128;
    int batchSize = 128;

    ReLUConfig lconfig(inputDim, batchSize);
    ReLULayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * inputDim);
    loadShareFromFile(std::string(TEST_PATH) + "test_relulayer_input", input);

    //comm_profiler.clear();
    //func_profiler.clear();
    //comm_profiler.start();
    //func_profiler.start();
    layer.forward(input);
    //func_profiler.dump_comm_rounds();
    //comm_profiler.dump_comm_bytes();

    // load expected output and compare with calculated activations
    LAYER_TEST_SHARE<T> expected(batchSize * inputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_relulayer_output", expected);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected);
    assertShare(*layer.getActivation(), host_expected);
}

TYPED_TEST(LayerTest, ReLUBackwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        -2, -3, 4, 3, 3.5, 1, -1.5, -1, 2, 1 << 12, -17, (1 << 18) - 1
    };

    ReLUConfig lconfig(
        input.size(),
        1 // "batch" size 
    );
    ReLULayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(input);

    LAYER_TEST_SHARE<T> delta = {
        -7.30e-1, -6.90e-2, -1.60e+0, -1.50e-1, -7.20e-1, 1.60e+0, 1.50e-1, 1.90e+0, -3.00e-1, -9.80e-1, 4.10e-1, -9.70e-1
    };

    layer.backward(delta, input);

    std::vector<double> expected = {
        0, 0, -1.60e+0, -1.50e-1, -7.20e-1, 1.60e+0, 0, 0, -3.00e-1, -9.80e-1, 0, -9.70e-1
    };
    assertShare(*layer.getDelta(), expected);
}

TYPED_TEST(LayerTest, MaxpoolForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // imageWidth x imageHeight = 2 x 2
    // features = 2
    // batchSize = 2
    // NHWC
    LAYER_TEST_SHARE<T> inputImage = {
        1, -1, -3, 2, 3, -2, 0, 1,
        2, -1, 0, 0, -2, -3, -1, 3
    };

    MaxpoolConfig lconfig(
        2, 2, // imageHeight x imageWidth
        2, // features
        2, // poolSize
        1, // stride
        2 // batchSize
    );
    MaxpoolLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(inputImage);

    std::vector<double> expected = {
        3, 2,
        2, 3
    };
    assertShare(*layer.getActivation(), expected);
}

TYPED_TEST(LayerTest, MaxpoolForwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int imageWidth = 9;
    int imageHeight = 9;
    int features = 2;
    int poolSize = 3;
    int stride = 2;
    int batchSize = 128;

    MaxpoolConfig lconfig(imageWidth, imageHeight, features, poolSize, stride, batchSize);
    MaxpoolLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * features * imageWidth * imageHeight);
    loadShareFromFile(std::string(TEST_PATH) + "test_maxpoollayer_input", input);

    //printShareTensor(input, "maxpool input", 1, 2, 9, 9);

    layer.forward(input);

    int outputWidth = ((imageWidth - poolSize)/stride) + 1;
    int outputHeight = ((imageHeight - poolSize)/stride) + 1;

    // load expected output and compare with calculated activations
    LAYER_TEST_SHARE<T> expected(batchSize * features * outputWidth * outputHeight);
    loadShareFromFile(std::string(TEST_PATH) + "test_maxpoollayer_output", expected);

    //printShareTensor(expected, "expected", 1, 2, 4, 4);

    std::vector<double> host_expected(expected.size());
    copyToHost(expected, host_expected);

    //printShareTensor(*layer.getActivation(), "activation", 1, 2, 4, 4);
    assertShare(*layer.getActivation(), host_expected);

    //printShareTensor(layer.maxPrime, "maxprime", 1, 2, 9, 9, false);
}

TYPED_TEST(LayerTest, MaxpoolBackwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // imageWidth x imageHeight = 2 x 2
    // features = 2
    // batchSize = 2
    // NHWC
    LAYER_TEST_SHARE<T> inputImage = {
        1, -1, -3, 2, 3, -2, 0, 1,
        2, -1, 0, 0, -2, -3, -1, 3 
    };

    MaxpoolConfig lconfig(
        2, 2, // imageWidth x imageHeight
        2, // features
        2, // poolSize
        1, // stride
        2 // batchSize
    );
    MaxpoolLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(inputImage);

    LAYER_TEST_SHARE<T> gradin = {
        1, 2, -1, -2
    };
    layer.backward(gradin, inputImage);

    // NHWC
    std::vector<double>expected = {
        0, 0, 0, 2, 1, 0, 0, 0,
        -1, 0, 0, 0, 0, 0, 0, -2
    };
    assertShare(*layer.getDelta(), expected);
}

TYPED_TEST(LayerTest, MaxpoolBackwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int imageWidth = 9;
    int imageHeight = 9;
    int features = 2;
    int poolSize = 3;
    int stride = 2;
    int batchSize = 128;

    MaxpoolConfig lconfig(imageWidth, imageHeight, features, poolSize, stride, batchSize);
    MaxpoolLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * features * imageWidth * imageHeight);
    loadShareFromFile(std::string(TEST_PATH) + "test_maxpoollayer_input", input);

    layer.forward(input);

    int outputWidth = ((imageWidth - poolSize)/stride) + 1;
    int outputHeight = ((imageHeight - poolSize)/stride) + 1;

    // load gradient in
    LAYER_TEST_SHARE<T> gradin(batchSize * features * outputWidth * outputHeight); 
    loadShareFromFile(std::string(TEST_PATH) + "test_maxpoollayer_gradin", gradin);

    layer.backward(gradin, input);

    // check matching gradout
    LAYER_TEST_SHARE<T> gradout(batchSize * features * imageWidth * imageHeight);
    loadShareFromFile(std::string(TEST_PATH) + "test_maxpoollayer_gradout", gradout);

    std::vector<double> expected_gradout(gradout.size());
    copyToHost(gradout, expected_gradout);
    assertShare(*layer.getDelta(), expected_gradout);
}

TYPED_TEST(LayerTest, AveragepoolForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // imageWidth x imageHeight = 2 x 2
    // features = 2
    // batchSize = 2
    // NHWC
    LAYER_TEST_SHARE<T> inputImage = {
        1, -1, -3, 2, 3, -2, 0, 1,
        2, -1, 0, 0, -2, -3, -1, 3
    };

    AveragepoolConfig lconfig(
        2, 2, // imageWidth x imageHeight
        2, // features
        2, // poolSize
        1, // stride
        2 // batchSize
    );
    AveragepoolLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(inputImage);

    std::vector<double> expected = {
        0.25, 0, -0.25, -0.25
    };
    assertShare(*layer.getActivation(), expected);
}

TYPED_TEST(LayerTest, AveragepoolBackwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // imageWidth x imageHeight = 2 x 2
    // features = 2
    // batchSize = 2
    // NHWC
    LAYER_TEST_SHARE<T> inputImage = {
        1, -1, -3, 2, 3, -2, 0, 1,
        2, -1, 0, 0, -2, -3, -1, 3
    };

    AveragepoolConfig lconfig(
        2, 2, // imageWidth x imageHeight
        2, // features
        2, // poolSize
        1, // stride
        2 // batchSize
    );
    AveragepoolLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(inputImage);

    LAYER_TEST_SHARE<T> gradin = {
        1, 0, -1, -2
    };
    layer.backward(gradin, inputImage);

    std::vector<double> expected = {
        0.25, 0.0,
        0.25, 0.0,
        0.25, 0.0,
        0.25, 0.0,
        -0.25, -0.50,
        -0.25, -0.50,
        -0.25, -0.50,
        -0.25, -0.50
    };
    assertShare(*layer.getDelta(), expected);
}

TYPED_TEST(LayerTest, CNNForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // B = 2, H = 3, W = 3, Din = 1
    LAYER_TEST_SHARE<T> im = {
        1, 2, 1, 2, 3, 2, 1, 2, 1,
        1, 2, 1, 2, 3, 2, 1, 2, 1,
    };

    // weights: 1 3x3 filters, Dout=1 -> 1 3x3 filter
    LAYER_TEST_SHARE<T> filter = {
        1, 0, 1, 0, 1, 0, 1, 0, 1
    };

    CNNConfig lconfig(
        3, 3, // image height x image width 
        1, // input features
        1, 3, // filters/output features, filter size
        1, 1, // stride, padding
        2 // batch size
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    layer.getWeights()->zero();
    *layer.getWeights() += filter;

    layer.forward(im);

    std::vector<double> expected = {
        4, 6, 4, 6, 7, 6, 4, 6, 4,
        4, 6, 4, 6, 7, 6, 4, 6, 4
    };

    assertShare(*layer.getActivation(), expected, true);
}

TYPED_TEST(LayerTest, CNNForwardStride) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // B = 1, H = 3, W = 3, Din = 1
    LAYER_TEST_SHARE<T> im = {
        1, 2, 1, 2, 3, 2, 1, 2, 1,
    };

    // weights: 1 3x3 filters, Dout=1 -> 1 3x3 filter
    LAYER_TEST_SHARE<T> filter = {
        1, 3, 1, 2, 1, 4, 1, 0, 1
    };

    CNNConfig lconfig(
        3, 3, // image height x image width 
        1, // input features
        1, 3, // filters/output features, filter size
        2, 1, // stride, padding
        1 // batch size
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    layer.getWeights()->zero();
    *layer.getWeights() += filter;

    layer.forward(im);

    std::vector<double> expected = {
        12, 8, 18, 14
    };

    assertShare(*layer.getActivation(), expected, true);
}

TYPED_TEST(LayerTest, CNNForwardInputChannels) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // B = 1, H = 3, W = 3, Din = 2
    LAYER_TEST_SHARE<T> im = {
        1, 0, 2, -1, 1, 3,
        2, 2, 3, 1, 2, 5,
        1, 6, 2, 0, 1, -2,

        1, 0, 2, -1, 1, 3,
        2, 2, 3, 1, 2, 5,
        1, 6, 2, 0, 1, -2
    };

    // weights: 2 3x3 filters, Dout=1 -> 2 3x3 filter
    // Dout x H x W x Din
    LAYER_TEST_SHARE<T> filter = {
        1, 1, 3, 1, 1, 1,
        2, 1, 1, 0, 4, -1,
        1, 0, 0, 1, 1, 0
    };

    CNNConfig lconfig(
        3, 3, // image height x image width 
        2, // input features
        1, 3, // filters/output features, filter size
        2, 1, // stride, padding
        2 // batch size
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    layer.getWeights()->zero();
    *layer.getWeights() += filter;

    layer.forward(im);

    std::vector<double> expected = {
        15, 12, 21, 20,
        15, 12, 21, 20
    };

    assertShare(*layer.getActivation(), expected, true);
}

TYPED_TEST(LayerTest, CNNForwardAllDims) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // N = 2, H = 3, W = 3, C = 2
    LAYER_TEST_SHARE<T> im = {
        1, -1, 1, 0, 2, 3,
        2, 0, 0, 2, 1, -1,
        1, -1, 3, 1, 0, 1,

        0, -1, -2, 2, -1, -1,
        0, 0, 0, 3, 2, -1,
        1, 1, 3, 0, 0, 3
    };

    // weights: 2 3x3 filters, Dout=2 -> 4 3x3 filter
    // Dout x H x W x Din
    LAYER_TEST_SHARE<T> filter = {
        1, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 1, 0,
        1, 0, 0, 1, 1, 1,

        -1, -1, 1, 0, -1, -1,
        1, 0, -1, -1, 1, 0,
        -1, -1, 1, 0, -1, -1
    };

    CNNConfig lconfig(
        3, 3, // image height x image width 
        2, // input features
        2, 3, // filters/output features, filter size
        2, 1, // stride, padding
        2 // batch size
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    layer.getWeights()->zero();
    *layer.getWeights() += filter;

    layer.forward(im);

    std::vector<double> expected = {
        2, 1, 3, -5,
        2, 3, 5, 1,

        0, -4, -4, -1,
        4, -2, 8, -1
    };

    assertShare(*layer.getActivation(), expected, true);
}

TYPED_TEST(LayerTest, CNNForwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputWidth = 9;
    int inputHeight = 9;
    int inputDim = 2;
    int outputDim = 3;
    int filterSize = 3;
    int stride = 2;
    int padding = 1;
    int batchSize = 128;

    CNNConfig lconfig(
        inputHeight, inputWidth,
        inputDim,
        outputDim, filterSize,
        stride, padding,
        batchSize
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    // load weights and biases
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_weight", *layer.getWeights());

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * inputWidth * inputHeight * inputDim);
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_input", input);

    //printShareTensor(input, "input", 1, 2, 9, 9);
    //printShareTensor(*layer.getWeights(), "weights", 1, 2, 3, 3);

    layer.forward(input);

    // load output
    LAYER_TEST_SHARE<T> expected(batchSize * 5 * 5 * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_output", expected);

    std::vector<double> expected_host(expected.size());
    copyToHost(expected, expected_host);
    assertShare(*layer.getActivation(), expected_host);
}

TYPED_TEST(LayerTest, CNNBackwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputWidth = 3;
    int inputHeight = 3;
    int inputDim = 2;
    int outputDim = 3;
    int filterSize = 3;
    int stride = 2;
    int padding = 1;
    int batchSize = 2;

    CNNConfig lconfig(
        inputWidth, inputHeight,
        inputDim,
        outputDim, filterSize,
        stride, padding,
        batchSize
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    LAYER_TEST_SHARE<T> weights = {
        -0.0017646551, 0.06236978, -0.1734625, -0.2251778, -0.004670009, 0.008731261, 0.12644096, -0.071232304, -0.09078175, -0.15610139, 0.18688585, 0.09318139, -0.1939936, -0.046330914, 0.06320529, -0.09716192, -0.020917177, 0.14142673, -0.15979224, -0.218654, 0.19572432, -0.09187673, -0.037991285, -0.108501494, -0.10263957, -0.14838351, -0.04850757, 0.20364694, 0.024940625, -0.16467114, 0.08561109, -0.059671625, 0.17637877, -0.1527774, 0.21342279, -0.22074954, -0.13758895, -0.121511586, 0.114238426, -0.10453278, 0.03987722, 0.23431943, 0.2026092, 0.14871348, 0.012395948, -0.008504704, -0.22007395, 0.09354602, 0.105174676, 0.13819723, -0.12084066, 0.15074591, -0.1703105, 0.03184168
    };
    layer.getWeights()->zero();
    *layer.getWeights() += weights;

    // load input
    LAYER_TEST_SHARE<T> input = {
        1.0, -2.0, 2.0, 1.0, 0.0, 3.0, 0.0, -1.0, 0.0, 2.0, 1.0, 4.0, 1.0, 0.0, -1.0, 1.0, 2.0, -1.0, 3.0, 1.0, -2.0, 0.0, -3.0, -1.0, -1.0, 1.0, 0.0, 2.0, -1.0, -4.0, 1.0, 0.0, 0.0, 3.0, -4.0, 4.0 
    };

    layer.forward(input);

    LAYER_TEST_SHARE<T> expected_activations = {
        1.0683895, -0.859313, -0.40425906, -0.70475876, -0.29679593, 1.2869551, 0.058154106, -0.3632456, 0.89918756, -1.1725695, -0.95549667, -0.56751496, -0.6797321, -0.7624123, 0.804101, 0.40834528, 0.4625425, -0.63965005, 0.15450963, -1.047125, 0.54290164, 0.7239378, 0.29794204, 0.42340738
    };
    assertShare(*layer.getActivation(), expected_activations);

    // load gradient in
    LAYER_TEST_SHARE<T> gradin = {
        3.0, 0.0, -1.0, 3.0, -1.0, 3.0, -4.0, 3.0, -2.0, -2.0, 2.0, 0.0, -2.0, 1.0, 0.0, -4.0, 2.0, 3.0, -4.0, 1.0, -3.0, 0.0, -4.0, 0.0
    };

    layer.backward(gradin, input);

    // check matching gradout
    LAYER_TEST_SHARE<T> gradout = {
        -0.2847412, -0.45979947, 1.8705214, 0.5668252, -0.18664983, -0.69746524, 1.3630025, 0.3919149, -0.7356124, -0.6634069, 0.38908872, 0.58013153, 0.19281238, 1.2523558, -0.69073474, -1.2081335, 0.084548354, 0.71949667, 0.13305593, 0.5158497, -0.4520465, 0.08326872, 0.3032997, 1.0061853, 0.5968272, 1.163979, 2.018204, 0.0052001923, -1.0454829, 0.9028375, 0.27743158, 0.85356665, 0.34817737, -0.22450069, 0.19403028, -0.8145878
    };

    assertShare(*layer.getDelta(), gradout);
}

TYPED_TEST(LayerTest, CNNBackwardMNIST) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    int inputWidth = 9;
    int inputHeight = 9;
    int inputDim = 2;
    int outputDim = 3;
    int filterSize = 3;
    int stride = 2;
    int padding = 1;
    int batchSize = 128;

    CNNConfig lconfig(
        inputWidth, inputHeight,
        inputDim,
        outputDim, filterSize,
        stride, padding,
        batchSize
    );
    CNNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    // load weights and biases
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_weight", *layer.getWeights());

    //printShareTensor(*layer.getWeights(), "weights", 3, 2, 3, 3);

    // load input
    LAYER_TEST_SHARE<T> input(batchSize * inputWidth * inputHeight * inputDim);
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_input", input);

    //printShareTensor(input, "inputs", 128, 2, 9, 9);

    // load gradient in
    LAYER_TEST_SHARE<T> gradin(batchSize * 5 * 5 * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_gradin", gradin);

    //printShareTensor(gradin, "gradin", 128, 3, 5, 5);

    layer.backward(gradin, input);

    // check matching gradout
    LAYER_TEST_SHARE<T> gradout(batchSize * inputWidth * inputHeight * inputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_gradout", gradout);

    //printShareTensor(*layer.getDelta(), "actual gradout", 128, inputDim, inputHeight, inputWidth);
    //printShareTensor(gradout, "expected gradout", 1, inputDim, inputHeight, inputWidth);

    std::vector<double> expected_gradout(gradout.size());
    copyToHost(gradout, expected_gradout);
    assertShare(*layer.getDelta(), expected_gradout);

    // check weight update
    LAYER_TEST_SHARE<T> new_weights(inputDim * filterSize * filterSize * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_newweight", new_weights);

    LAYER_TEST_SHARE<T> grad_weights(inputDim * filterSize * filterSize * outputDim); 
    loadShareFromFile(std::string(TEST_PATH) + "test_convlayer_gradweight", grad_weights);

    std::vector<double> expected_new_weights(new_weights.size());
    copyToHost(new_weights, expected_new_weights);
    assertShare(*layer.getWeights(), expected_new_weights);
}

TYPED_TEST(LayerTest, LNForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        1, 2, 3, 4, -1, -2, 0, 1
    };

    LNConfig lconfig(
        input.size(),
        1 // batch size
    );
    LNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(input);

    std::vector<double> expected = {
        0, 0.534522, 1.069044, 1.603567, -1.069044, -1.603567, -0.534522, 0
    };
    assertShare(*layer.getActivation(), expected, true, 1e-2);
}

TYPED_TEST(LayerTest, LNForwardBasic2) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        1, 2, 3, 4, -1, -2, 0, 1,
        4, 2, 3, 1, -1, 1, 0, -2,
    };

    int batches = 2;
    LNConfig lconfig(
        input.size() / batches,
        batches // batch size
    );
    LNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(input);

    std::vector<double> expected = {
        0, 0.534522, 1.069044, 1.603567, -1.069044, -1.603567, -0.534522, 0,
        1.603567, 0.534522, 1.069044, 0, -1.069044, 0, -0.534522, -1.603567
    };
    assertShare(*layer.getActivation(), expected, true, 1e-2);
}

TYPED_TEST(LayerTest, LNForward) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        -0.16704008  , -0.23331553  ,  0.055922203 , -0.2650258   , -0.3377273   , -0.12555759  , -0.17132029  , -0.27296346  ,  0.14300632  , -0.14093764  , -0.10383279  ,  0.19090839  , -0.04390003  , -0.21765414  ,  0.007576036 , -0.05555154  , -0.36651954  , -0.0073424797, -0.109995134 , -0.14744322  ,  0.20732933  , -0.19366084  , -0.12744121  ,  0.25312233  , -0.1431271   , -0.32392293  ,  0.05132977  ,  0.0610541   , -0.30057257  ,  0.18669829  , -0.010873071 ,  0.0042723576,  0.39242274  , -0.06731775  ,  0.040623672 ,  0.27358758  , -0.23122413  , -0.2865307   ,  0.0093831485,  0.035033435 , -0.09664877  ,  0.18795612  ,  0.008636649 ,  0.043175384 ,  0.39666653  ,  0.11124479  ,  0.10529415  ,  0.34064227  , -0.112778366 ,  0.12296915  ,  0.1013414   , -0.05720646  ,  0.011465586 ,  0.13785137  ,  0.010274639 , -0.10554777  ,  0.13159882  , -0.029448895 , -0.048179436 ,  0.21179217  , -0.030752283 ,  0.123240635 ,  0.07087865  ,  0.02148806  , -0.20858702  ,  0.18253933  , -0.10028666  , -0.30243334  ,  0.25251052  , -0.044802513 , -0.10984374  ,  0.24827938  , -0.0925099   ,  0.0004889969,  0.007567016 , -0.06872087  , -0.2129896   ,  0.077127874 , -0.1984907   , -0.10820356  ,  0.12079788  , -0.05359043  ,  0.044336077 ,  0.16279252  , -0.22533682  , -0.3979688   , -0.046249334 , -0.13013805  , -0.3319337   , -0.0020106707,  0.03863339  , -0.19886573  ,  0.1434916   ,  0.1468901   ,  0.053208254 ,  0.17777258
    };

    log_learning_rate = 3;
    int batches = 2;
    LNConfig lconfig(
        input.size() / batches,
        batches // batch size
    );
    LNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);
    layer.forward(input);

    std::vector<double> expected = {
        -0.6754183  , -1.0021422  ,  0.42373848 , -1.1584672  , -1.5168701  , -0.47091842 , -0.69651884 , -1.1975981  ,  0.85304475 , -0.5467388  , -0.36381978 ,  1.0891918  , -0.06836398 , -0.92493486 ,  0.18540213 , -0.12580346 , -1.6588098  ,  0.111857   , -0.39419883 , -0.57881    ,  1.1701436  , -0.806653   , -0.48020428 ,  1.3958933  , -0.55753237 , -1.4488175  ,  0.40109873 ,  0.44903767 , -1.3337052  ,  1.0684369  ,  0.094451934,  0.16911569 ,  2.0826147  , -0.18380837 ,  0.34831995 ,  1.4967827  , -0.99183214 , -1.2644818  ,  0.1943108  ,  0.32076126 , -0.32840407 ,  1.0746378  ,  0.19063072 ,  0.36089936 ,  2.1035357  ,  0.6964671  ,  0.6671317  ,  1.8273481  , -0.61330867 ,  0.83381057 ,  0.70105016 , -0.2721845  ,  0.14935394 ,  0.925164   ,  0.1420434  , -0.5689242  ,  0.8867831  , -0.10179666 , -0.21677274 ,  1.3790443  , -0.10979742 ,  0.835477   ,  0.51405674 ,  0.21087617 , -1.2014232  ,  1.1994779  , -0.5366292  , -1.777492   ,  1.628991   , -0.19604376 , -0.5952946  ,  1.6030184  , -0.4888921  ,  0.08197494 ,  0.12542285 , -0.34286487 , -1.228448   ,  0.55241716 , -1.1394476  , -0.58522654 ,  0.8204824  , -0.24998775 ,  0.3511271  ,  1.0782634  , -1.3042406  , -2.3639295  , -0.20492494 , -0.71986985 , -1.9585779  ,  0.06663091 ,  0.31612158 , -1.1417497  ,  0.9597862  ,  0.98064756 ,  0.4055883  ,  1.1702175
    };

    //printShare(*layer.getActivation(), "activations");
    assertShare(*layer.getActivation(), expected, true, 1e-2);
}

TYPED_TEST(LayerTest, LNBackwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        1, 2, 3, 4, -1, -2, 0, 1
    };

    LNConfig lconfig(
        input.size(),
        1 // batch size
    );
    LNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);

    layer.forward(input);

    LAYER_TEST_SHARE<T> delta = {
        -0.1901, 0.9232, 0.0208, -0.5175, -0.4921, 0.7767, -0.0701, 0.5335 
    };
    layer.backward(delta, input);

    std::vector<double> expected = {
        -0.16738572,  0.46327254,  0.01649384, -0.23566524, -0.39996027, 0.24266748, -0.1388174 ,  0.21939475     
    };
    assertShare(*layer.getDelta(), expected, true, 1e-2);
}

TYPED_TEST(LayerTest, LNBackward) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    LAYER_TEST_SHARE<T> input = {
        -0.16704008  , -0.23331553  ,  0.055922203 , -0.2650258   , -0.3377273   , -0.12555759  , -0.17132029  , -0.27296346  ,  0.14300632  , -0.14093764  , -0.10383279  ,  0.19090839  , -0.04390003  , -0.21765414  ,  0.007576036 , -0.05555154  , -0.36651954  , -0.0073424797, -0.109995134 , -0.14744322  ,  0.20732933  , -0.19366084  , -0.12744121  ,  0.25312233  , -0.1431271   , -0.32392293  ,  0.05132977  ,  0.0610541   , -0.30057257  ,  0.18669829  , -0.010873071 ,  0.0042723576,  0.39242274  , -0.06731775  ,  0.040623672 ,  0.27358758  , -0.23122413  , -0.2865307   ,  0.0093831485,  0.035033435 , -0.09664877  ,  0.18795612  ,  0.008636649 ,  0.043175384 ,  0.39666653  ,  0.11124479  ,  0.10529415  ,  0.34064227  , -0.112778366 ,  0.12296915  ,  0.1013414   , -0.05720646  ,  0.011465586 ,  0.13785137  ,  0.010274639 , -0.10554777  ,  0.13159882  , -0.029448895 , -0.048179436 ,  0.21179217  , -0.030752283 ,  0.123240635 ,  0.07087865  ,  0.02148806  , -0.20858702  ,  0.18253933  , -0.10028666  , -0.30243334  ,  0.25251052  , -0.044802513 , -0.10984374  ,  0.24827938  , -0.0925099   ,  0.0004889969,  0.007567016 , -0.06872087  , -0.2129896   ,  0.077127874 , -0.1984907   , -0.10820356  ,  0.12079788  , -0.05359043  ,  0.044336077 ,  0.16279252  , -0.22533682  , -0.3979688   , -0.046249334 , -0.13013805  , -0.3319337   , -0.0020106707,  0.03863339  , -0.19886573  ,  0.1434916   ,  0.1468901   ,  0.053208254 ,  0.17777258
    };

    log_learning_rate = 3;

    int batches = 2;
    LNConfig lconfig(
        input.size() / batches,
        batches
    );
    LNLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef);

    layer.forward(input);

    LAYER_TEST_SHARE<T> delta = {
        -0.003016197 ,  0.04465534  ,  0.0037896447, -0.033749204 ,  0.0066000246,  0.01155613  , -0.019346703 ,  0.042821195 , -0.027043447 ,  0.022255544 , -0.04911209  , -0.040082216 , -0.037328377 , -0.05257369  ,  0.043693684 ,  0.01958153  ,  0.0061617596,  0.06208346  , -0.020970399 ,  0.023291297 , -0.02565262  , -0.056029744 ,  0.048401937 , -0.05465204  , -0.016840862 ,  0.06484781  , -0.072117776 ,  0.04595969  ,  0.046031658 ,  0.03249733  , -0.0014214292, -0.027930401 ,  0.026949331 ,  0.04652488  ,  0.00750748  , -0.05216788  , -0.05130661  , -0.004564652 ,  0.010234315 , -0.014690556 ,  0.06012882  , -0.017501594 ,  0.07179859  ,  0.026257886 ,  0.027632989 , -0.05186768  ,  0.000330783 ,  0.01475822  ,  0.03132531  , -0.014807506 ,  0.019777738 ,  0.04680904  ,  0.036492724 , -0.050003912 ,  0.07108652  ,  0.028978072 ,  0.04177461  , -0.013561761 ,  0.04260288  , -0.0020226175,  0.059592914 ,  0.056133546 ,  0.01499483  , -0.010970011 , -0.06396218  , -0.06649235  ,  0.003923188 ,  0.045088723 ,  0.026736949 ,  0.037610114 ,  0.0022881208, -0.029044    , -0.015213385 , -0.012955584 , -0.021162214 ,  0.014669538 ,  0.0034637162, -0.0124046095,  0.027675144 , -0.007870437 ,  0.007155791 ,  0.053678486 , -0.048064183 ,  0.00905535  ,  0.07109181  , -0.0320914   ,  0.052838594 , -0.021588517 , -0.0658681   ,  0.0062944293, -0.023589961 , -0.03572181  , -0.047861855 ,  0.022831915 ,  0.03108764  , -0.012535142
    };
    layer.backward(delta, input);

    std::vector<double> expected = {
        -0.03878477  ,  0.18894875  ,  0.019247893 , -0.20105085  , -0.010120329 ,  0.0376085   , -0.11976067  ,  0.17555344  , -0.123190865 ,  0.088665605 , -0.25908756  , -0.18220958  , -0.19441575  , -0.28865016  ,  0.21065794  ,  0.08485869  , -0.015442275 ,  0.29967758  , -0.121031635 ,  0.093057334 , -0.109271705 , -0.30305332  ,  0.2190437   , -0.24720462  , -0.104311794 ,  0.27854463  , -0.35546386  ,  0.22770076  ,  0.18834886  ,  0.17512995  , -0.013775731 , -0.14279638  ,  0.17036808  ,  0.21639177  ,  0.035896253 , -0.23271115  , -0.28389367  , -0.05953842  ,  0.045908753 , -0.07414918  ,  0.28023577  , -0.071216054 ,  0.34932563  ,  0.12861195  ,  0.17420432  , -0.24905647  ,  0.0076174885,  0.104582965 ,  0.15692632  , -0.120448336 ,  0.09131785  ,  0.25334126  ,  0.19170722  , -0.3361323   ,  0.40402943  ,  0.14269611  ,  0.22708954  , -0.11655668  ,  0.22774445  , -0.03978079  ,  0.33246595  ,  0.3150249   ,  0.06120777  , -0.09939243  , -0.4303496   , -0.43624437  , -0.01097187  ,  0.23673935  ,  0.13776095  ,  0.19717991  , -0.02124408  , -0.20475064  , -0.12824875  , -0.1120981   , -0.16229953  ,  0.055771537 , -0.016569074 , -0.10682776  ,  0.13240826  , -0.08356124  ,  0.014318299 ,  0.29559797  , -0.32652944  ,  0.027013265 ,  0.3982569   , -0.23937911  ,  0.2906232   , -0.1683091   , -0.44508803  ,  0.006005142 , -0.17643666  , -0.25675863  , -0.32284445  ,  0.11118792  ,  0.15955697  , -0.105149336 
    };
    //printShare(*layer.getDelta(), "delta");
    assertShare(*layer.getDelta(), expected, true, 1e-2);

    std::vector<double> expected_gamma = {
        1.0021468 , 1.0071372 , 0.9980661 , 0.9967054 , 1.0005702 , 1.0064629 , 0.9970534 , 1.0084711 , 0.99825305, 1.0013484 , 0.9989209 , 1.0058059 , 1.0004989 , 0.9880593 , 0.99802387, 1.0005971 , 0.9916719 , 1.0091015 , 0.99922985, 1.0117033 , 0.9983079 , 0.9952721 , 1.0030756 , 1.0153558 , 0.9978966 , 1.0118768 , 1.0039476 , 0.998049  , 1.008206  , 0.9965164 , 1.0039586 , 1.0000147 , 0.99225044, 1.0027463 , 1.0017827 , 1.00854   , 1.0052291 , 0.9897958 , 1.001105  , 0.9986464 , 0.9863424 , 1.0022986 , 0.99922127, 0.99371725, 0.99847627, 1.0017167 , 0.99839634, 0.99846256
    };
    //printShare(*layer.getWeights(), "gamma");
    assertShare(*layer.getWeights(), expected_gamma);

    std::vector<double> expected_beta = {
        -0.0035386393, -0.0037309793, -0.0029459228, -0.0016324795, -0.0053865938,  0.004805973 , -0.006467477 , -0.008974908 , -0.0018413952, -0.0010867228,  0.0008136509,  0.005263104 , -0.002783067 , -0.0004449822, -0.0073360642, -0.0010764399,  0.007225052 ,  0.0005511111,  0.0021309014, -0.008547503 , -0.0001355412,  0.0023024539, -0.0063362573,  0.010462005 ,  0.0040067807, -0.0064865286,  0.011659998 , -0.0075786533, -0.0061869216, -0.0025115903, -0.0032817144,  0.0044751046, -0.0042631403, -0.012525421 ,  0.005069588 ,  0.0053890664, -0.00247315  ,  0.0045820065, -0.007884113 ,  0.0045348844,  0.0007174104,  0.0014008956, -0.006026079 ,  0.0011829904,  0.0025286083,  0.0036294705, -0.003927303 , -0.0002778848
    };
    //printShare(*layer.getBiases(), "beta");
    assertShare(*layer.getBiases(), expected_beta);
}

TYPED_TEST(LayerTest, ResForwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // B = 2, H = 3, W = 3, Din = 2
    LAYER_TEST_SHARE<T> im = {
        -0.3433358 ,-0.5341458 , 0.37994185,-0.8869632 ,-0.28129023,-1.7610981 , 1.5712649 ,-0.523379  ,-0.14475703, 0.8387747 ,-1.3298751 ,-1.4777428 , 0.19161488, 0.86150426, 0.6376178 , 1.1528951 ,-0.1420108 ,-1.7556711 , 0.07616609, 1.361524  ,-1.1560066 , 1.6952796 , 1.4902748 ,-0.5650247 ,-1.0786034 ,-0.3660986 ,-0.14336488, 2.0654995 ,-0.7005269 , 0.927811  , 0.5851132 ,-1.3270525 ,-0.1947406 , 0.25783238, 0.18056405, 0.4825783
    };

    LAYER_TEST_SHARE<T> conv1_filter = {
        -0.00176466, 0.06236978,-0.1734625 ,-0.2251778 ,-0.00467001, 0.00873125, 0.12644097,-0.0712323 ,-0.09078176,-0.15610139, 0.18688585, 0.0931814 ,-0.1939936 ,-0.04633091, 0.06320529,-0.09716192,-0.02091717, 0.14142673,-0.15979224,-0.218654  , 0.19572432,-0.09187673,-0.03799128,-0.1085015 ,-0.10263957,-0.14838351,-0.04850757, 0.20364694, 0.02494062,-0.16467114, 0.0856111 ,-0.05967162, 0.17637877,-0.1527774 , 0.21342279,-0.22074954,-0.13758895,-0.12151159, 0.11423842,-0.10453279, 0.03987721, 0.23431942, 0.2026092 , 0.1487135 , 0.01239595,-0.0085047 ,-0.22007395, 0.09354603, 0.10517468, 0.13819724,-0.12084066, 0.1507459 ,-0.1703105 , 0.03184169, 0.15803514, 0.07120732,-0.18274125, 0.00899948, 0.10664892, 0.22631992,-0.13878204, 0.12939395,-0.16336197, 0.05461333, 0.09479012,-0.18163772, 0.04392171,-0.02974972,-0.12175991, 0.14622417,-0.13961883,-0.08637775
    };

    LAYER_TEST_SHARE<T> conv2_filter = {
        0.06550165, 0.14808208, 0.06109856,-0.09596992, 0.14705947, 0.03119534, 0.06231853,-0.07267086, 0.01533208,-0.07629281, 0.08321917,-0.00299569, 0.13809156, 0.12672663, 0.08428469, 0.15679155, 0.03316909,-0.02807667,-0.16495588,-0.04194747,-0.10426756, 0.0640927 , 0.03488356,-0.12550983, 0.14503455,-0.16625467, 0.11931193, 0.11230298,-0.14493045,-0.02742686,-0.10811615,-0.15876636,-0.1553257 ,-0.09871726,-0.130014  ,-0.12855946,-0.00918327, 0.00752044,-0.04172164,-0.01713338, 0.09889627, 0.06540471,-0.13660362, 0.03373023, 0.11421665, 0.07886484,-0.10857763, 0.10584265, 0.02502418, 0.02431685,-0.08118556, 0.00463204,-0.10142319, 0.00998336,-0.03545253, 0.10597324,-0.14054716,-0.15987483,-0.00855323, 0.1582356 ,-0.06825505, 0.03952905,-0.05830556,-0.01437815, 0.15122835,-0.08132146, 0.03562608, 0.15787438,-0.04148072,-0.09878445, 0.11930847,-0.01205361,-0.14972025, 0.091401  ,-0.14955834,-0.067982  ,-0.00108041, 0.10396367, 0.00215536,-0.15041016,-0.15597534,-0.06757878,-0.15000974,-0.07440114,-0.07901287, 0.09010684,-0.14390633, 0.15050052,-0.08284106,-0.13041824,-0.07571249, 0.10544956,-0.14066803,-0.03210255,-0.01124787, 0.13327555, 0.1134842 ,-0.16073865,-0.02607991, 0.06035896,-0.12771863,-0.03523505, 0.0627832 ,-0.01923241,-0.03380613,-0.03272378, 0.14656991,-0.13468254, 0.01788416, 0.09030735,-0.09931886, 0.14058146, 0.04652409,-0.11756305, 0.10743718,-0.00124067,-0.04007912,-0.09174737,-0.09677693,-0.03805745,-0.03489479, 0.13232969,-0.0036362 , 0.06026793, 0.08008423, 0.00743332,-0.12598668,-0.03294693,-0.03505091,-0.09711759, 0.11665557,-0.00117107, 0.11901878, 0.11403748, 0.00701135, 0.02110434, 0.05885983,-0.11748902,-0.11441871, 0.02091106,-0.13734691, 0.05695751,-0.05991083, 0.02126551
    };

    LAYER_TEST_SHARE<T> conv_shortcut_filter = {
        -0.55310255,-0.37061688, 0.5709836 ,-0.57385015,-0.05077465, 0.6994974 , 0.25543317, 0.02002034
    };

    ResLayerConfig lconfig(
        2, 3, 3, // batch size, height, width
        2, 4, 1, 2, 1 // in planes, planes, num blocks, stride, expansion
    );
    ResLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    //layer.printLayer();

    // vector of layer pointers
    auto blk = *layer.getBlock(0);
    auto shortcut = *layer.getShortcut(0);

    // CNN 1
    blk[0]->getWeights()->zero();
    *blk[0]->getWeights() += conv1_filter;

    // CNN 2
    blk[3]->getWeights()->zero();
    *blk[3]->getWeights() += conv2_filter;

    // CNN shortcut
    shortcut[0]->getWeights()->zero();
    *shortcut[0]->getWeights() += conv_shortcut_filter;

    layer.forward(im);

    std::vector<double> expected = {
        0.        ,0.        ,0.        ,0.        ,0.9588918 ,2.7850027 ,0.        ,1.1165558 ,0.        ,0.        ,0.        ,0.00948378,2.7812614 ,1.4477533 ,0.        ,1.1872882,0.        ,0.41236782,0.95787394,0.1546284 ,0.        ,1.7484272 ,0.        ,0.7467702 ,0.1647719 ,1.4662426 ,0.        ,0.09182754,1.481421  ,0.        ,0.        ,0.31312495
    };

    assertShare(*layer.getActivation(), expected, true);
}

TYPED_TEST(LayerTest, ResBackwardBasic) {

    using T = typename TestFixture::ParamType;

    if (partyNum >= LAYER_TEST_SHARE<T>::numParties) return;

    // B = 2, H = 3, W = 3, Din = 2
    LAYER_TEST_SHARE<T> im = {
        -0.42623055,-0.68835765, 0.612481  ,-0.53281415, 0.23289491, 0.31327626, 1.3317285 ,-0.45654368,-0.15132089, 0.07914637, 0.70316255,-0.42534888, 1.0588793 , 0.38239938, 0.73966444, 0.55038875, 0.57641155, 0.40002853, 0.2707914 , 0.49398395,-0.04872167, 0.27918553,-1.0344548 , 0.4968742 , 0.9972433 , 0.7366074 ,-0.46596402,-0.35968798,-0.16035673, 0.1397571 ,-0.29486123,-0.13685007, 0.70072854, 0.7624139 , 0.39911157,-0.61572266
    };

    LAYER_TEST_SHARE<T> conv1_filter = {
        -0.18136837, 0.21662165,-0.02682591, 0.2255866 , 0.15072899, 0.1325011 , 0.15823407, 0.09638824,-0.1363976 , 0.17898612, 0.0175207 ,-0.13392344, 0.16756809,-0.17893228, 0.18217556,-0.08591132,-0.11128336,-0.03693792, 0.20013507,-0.145532  ,-0.07877946, 0.1169178 , 0.02255685,-0.17516084, 0.00973901, 0.01208075,-0.06398024,-0.21541737, 0.21798442,-0.10057075,-0.16669293, 0.11301621,-0.04547375,-0.04217714, 0.01261958, 0.08492196,-0.1673793 ,-0.21398692, 0.01545932,-0.04781535, 0.05146892, 0.19591269, 0.08761686,-0.22657901,-0.15709266, 0.15849395,-0.17968008,-0.09428629, 0.20005892,-0.2290226 ,-0.08444858,-0.22308724, 0.11709931, 0.06903344, 0.01074871, 0.11177389, 0.12691234,-0.23143077,-0.15557976,-0.18506439,-0.21253723,-0.18899293, 0.23428702,-0.09180427, 0.19671328, 0.07513344, 0.19547473,-0.06779422, 0.11907966, 0.05084832, 0.01266768, 0.12652658
    };

    LAYER_TEST_SHARE<T> conv2_filter = {
        0.02321821,-0.16351049,-0.01698252,-0.11676499,-0.05141944,-0.09440254, 0.09222376,-0.1396576 , 0.09583817,-0.08255182,-0.03623861, 0.1524212 ,-0.1115139 ,-0.14235023, 0.12826686, 0.13987964, 0.07315969, 0.08018261, 0.00535935,-0.09017631,-0.01876821,-0.1372815 , 0.0221501 ,-0.15437995,-0.12921995, 0.07776799, 0.10314536,-0.01812168, 0.16439942,-0.11765536,-0.05152967, 0.14746962, 0.05843608, 0.08697345, 0.08261826, 0.11754972, 0.08352654, 0.10304356, 0.05072232,-0.08074926,-0.08982506, 0.04195818,-0.12073237,-0.12569778,-0.0492489 , 0.01473306, 0.15316848,-0.00080379, 0.09865308,-0.04624244,-0.04019783, 0.16638169, 0.05262929, 0.05911561,-0.0980048 ,-0.13511166, 0.05577552, 0.09658929,-0.04484735,-0.04248399, 0.14108822,-0.06213262, 0.0584166 , 0.16277836, 0.06820554,-0.08142861,-0.08459842,-0.12633216,-0.04795223,-0.01658279,-0.00045506,-0.10909095,-0.05977869, 0.00773907,-0.0351095 , 0.15715328, 0.03693042,-0.15812722, 0.00796616, 0.130593  , 0.02289758,-0.10544833,-0.05681672, 0.0964074 , 0.03148848,-0.06146129,-0.03722463,-0.03712156,-0.03821931,-0.15977687,-0.1622691 , 0.08376795, 0.13704304, 0.03195389, 0.0838694 ,-0.05055898,-0.08708179, 0.16355106, 0.10590677,-0.029927  ,-0.08076056, 0.16423082,-0.09839337, 0.1413534 ,-0.11267988,-0.01438435,-0.10785562,-0.11057874,-0.01240051, 0.04176657,-0.15060386,-0.06288724,-0.15456167, 0.03368865, 0.07265123,-0.07161333,-0.0474472 , 0.12939703, 0.15789053,-0.07709873, 0.1379396 ,-0.01261995, 0.15561494,-0.1146273 , 0.06832141, 0.06329314,-0.14420351,-0.07609242,-0.13800393,-0.02494729, 0.15047672,-0.08208703,-0.05593663,-0.08416454, 0.07367587, 0.1578276 , 0.16224524, 0.13255316, 0.15433596, 0.08731882,-0.15117662,-0.1469612 ,-0.1405975 ,-0.01457858
    };

    LAYER_TEST_SHARE<T> conv_shortcut_filter = {
        -0.06796677,-0.55082095, 0.5894924 ,-0.31192014, 0.2508213 , 0.6150299 , 0.3566056 , 0.10011208
    };

    ResLayerConfig lconfig(
        2, 3, 3, // batch size, height, width
        2, 4, 1, 2, 1 // in planes, planes, num blocks, stride, expansion
    );
    ResLayer<T, LAYER_TEST_SHARE> layer(&lconfig, 0, 0xbeef); 

    //layer.printLayer();

    // vector of layer pointers
    auto blk = *layer.getBlock(0);
    auto shortcut = *layer.getShortcut(0);

    // CNN 1
    blk[0]->getWeights()->zero();
    *blk[0]->getWeights() += conv1_filter;

    // CNN 2
    blk[3]->getWeights()->zero();
    *blk[3]->getWeights() += conv2_filter;

    // CNN shortcut
    shortcut[0]->getWeights()->zero();
    *shortcut[0]->getWeights() += conv_shortcut_filter;

    layer.forward(im);

    std::vector<double> expected = {
        2.2812853 ,0.        ,0.        ,0.        ,0.        ,1.2604135 ,0.        ,1.2511618 ,0.        ,0.02690899,1.6001879 ,0.        ,0.        ,1.601757  ,0.68509704,0.        ,0.        ,0.        ,3.5309653 ,0.827518  ,0.        ,0.        ,0.24470696,0.20720953,0.        ,0.        ,0.0489687 ,0.        ,1.3336323 ,1.7274061 ,0.        ,0.
    };
    assertShare(*layer.getActivation(), expected, true);

    LAYER_TEST_SHARE<T> grad_wrt_output = {
        0.00167308, 0.03249155,-0.01799875, 0.10715706, 0.0482621 , 0.09416154, 0.04022709, 0.13324857, 0.02699861,-0.06481859, 0.00251701,-0.16047508, 0.13053599, 0.00793497,-0.07400178, 0.02706433,-0.09338936,-0.13583744, 0.06607803,-0.06850259,-0.01919051,-0.10653094, 0.03581957,-0.02655043, 0.05780746,-0.13130166, 0.07376945, 0.05819375, 0.01266077, 0.08955093, 0.00977797, 0.03723706
    };

    layer.backward(grad_wrt_output, im);

    std::vector<double> expected_delta = {
        -0.10737018, 0.00872805,-0.05043619, 0.06225888, 0.33844084,-0.04951637, 0.02141629,-0.02330437,-0.10192778, 0.05693968, 0.04599799, 0.10499842,-0.08968552, 0.1045114 , 0.03668717, 0.0192065 ,-0.07213426,-0.08027506,-0.03790957, 0.02118296, 0.01864827, 0.12390085, 0.08566047, 0.00448427, 0.02676383,-0.01236064, 0.01684967, 0.0774281 , 0.02158924, 0.15261975, 0.05943494, 0.20746344,-0.00592127, 0.03870514, 0.05555319,-0.09975185 
    };
    assertShare(*layer.getDelta(), expected_delta, true);
}

