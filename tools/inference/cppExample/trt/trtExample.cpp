#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <string>

class TRTLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            switch (severity) {
                case Severity::kINFO:
                std::cout<<msg<<std::endl;
                break;
                case Severity::kWARNING:
                std::cout<<msg<<std::endl;
                break;
                case Severity::kERROR:
                std::cout<<msg<<std::endl;
                break;
                case Severity::kINTERNAL_ERROR:
                std::cout<<msg<<std::endl;
                break;
                // case Severity::kVERBOSE:
                // std::cout<<msg<<std::endl;
                // break;
                default:
                break;
            }
        }
        static TRTLogger& getInstance() {
            static TRTLogger trt_logger{};
            return trt_logger;
        }
};

void main(){
    cv::Mat imageMat = cv::imread("your/png/path");
    cv::Mat inferMat;
    cv::dnn::blobFromImage(imageMat,inferMat,1.0 / 255.0);
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    std::ifstream ifs("your/trt/model/path", std::ios::binary | std::ios::in);
    if (!ifs.is_open())
        return;
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string str(size, '\0');
    ifs.read(str.data(), size);
    runtime = nvinfer1::createInferRuntime(TRTLogger::getInstance());
    engine = runtime->deserializeCudaEngine(str.data(), str.size());
    auto num = engine->getNbIOTensors();
    for (int32_t i = 0; i < num; i++){
        std::cout<<engine->getIOTensorName(i);
        auto shape = engine->getTensorShape(engine->getIOTensorName(i));
        std::cout<<"Tensor shape is :";
        for (auto j = 0; j < shape.nbDims; j++){
            std::cout<<shape.d[j]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"Tensor format description is :"<<engine->getTensorFormatDesc(engine->getIOTensorName(i))<<std::endl;
    }
    // change the data type for your model.
    float* image = nullptr;
    int64_t* imageSize = nullptr;
    int64_t* label = nullptr;
    float* score = nullptr;
    float* boxes = nullptr;
    // change the shape for your setting.
    cudaMalloc((void**)&image, 1 * 3 * 640 * 640 * sizeof(float));
    cudaMalloc((void**)&imageSize, 2 * sizeof(int64_t));
    cudaMalloc((void**)&label, 1 * 300 * sizeof(int64_t));
    cudaMalloc((void**)&boxes, 1 * 4 * 300 * sizeof(float));
    cudaMalloc((void**)&score, 1 * 300 * sizeof(float));
    cudaMemcpy(image, inferMat.ptr<float>(), inferMat.total() * sizeof(float), cudaMemcpyHostToDevice);
    int64_t h_inputData2[2] = {640, 640};
    cudaMemcpy(imageSize, h_inputData2, 2 * sizeof(int64_t), cudaMemcpyHostToDevice);

    void* buffers[5];
    buffers[0] = image;
    buffers[1] = imageSize;
    buffers[2] = label;
    buffers[3] = boxes;
    buffers[4] = score;

    auto context = engine->createExecutionContext();
    context->setInputShape(engine->getIOTensorName(0), nvinfer1::Dims4(1, 3, 640, 640));
    context->setInputShape(engine->getIOTensorName(1), nvinfer1::Dims2(1, 2));
    context->executeV2(buffers);

    float* hostScore = new float[1 * 300];
    int64_t* hostLabel = new int64_t[1 * 300];
    float* hostBoxes = new float[1 * 4 * 300];

    cudaMemcpy(hostScore, score, 1 * 300 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostLabel, label, 1 * 300 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBoxes, boxes, 1 * 4 * 300 *sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<2;i++){
        std::cout<<"score:"<<hostScore[i]<<" ";
        std::cout<<"label:"<<hostLabel[i]<<" ";
        std::cout<<"x1:"<<hostBoxes[4*i]<<" ";
        std::cout<<"y1:"<<hostBoxes[4*i+1]<<" ";
        std::cout<<"x2:"<<hostBoxes[4*i+2]<<" ";
        std::cout<<"y2:"<<hostBoxes[4*i+3]<<std::endl;
        auto cx = hostBoxes[4*i];
        auto cy = hostBoxes[4*i+1];
        auto bx = hostBoxes[4*i+2];
        auto by = hostBoxes[4*i+3];
        cv::rectangle(imageMat, cv::Rect2f(cx,cy, bx-cx, by-cy), cv::Scalar(0, 255, 0), 1);
    }
    cv::imwrite("your/save/path",imageMat);
    cudaFree(image);
    cudaFree(imageSize);
    cudaFree(label);
    cudaFree(score);
    cudaFree(boxes);
    delete [] hostScore;
    delete [] hostLabel;
    delete [] hostBoxes;
    delete context;
    delete engine;
    delete runtime;
    std::cout<<"finish"<<std::endl;
}
