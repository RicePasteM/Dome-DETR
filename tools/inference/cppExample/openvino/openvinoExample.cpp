#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/dnn.hpp>

int main(){
    try{
        ov::Core core;
        ov::CompiledModel mCompiledModel = core.compile_model("DOME.onnx","AUTO");
        cv::Mat imageMat = cv::imread("test.png");
        cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
        cv::Mat inferMat;
        cv::dnn::blobFromImage(imageMat,inferMat,1.0 / 255.0);
        std::vector<float> im_shape = { (float)1/imageMat.rows, (float)1/imageMat.cols };
        auto ireq = mCompiledModel.create_infer_request();
        auto inputports = mCompiledModel.inputs();
        ov::Tensor input_tensor1(inputports[0].get_element_type(), { 1,3,640,640 }, inferMat.ptr());
        ireq.set_input_tensor(0,input_tensor1);
        ov::Tensor input_tensor2(inputports[1].get_element_type(), { 1,2 });
        int64* input_tensor_data = input_tensor2.data<int64>();
        for (int i = 0; i < 2; i++) {
            input_tensor_data[i] = 640;
        }
        ireq.set_input_tensor(1,input_tensor2);
        ireq.infer();
        ov::Tensor labels_tensor = ireq.get_output_tensor(0);
        ov::Tensor bboxs_tensor = ireq.get_output_tensor(1);
        ov::Tensor scores_tensor = ireq.get_output_tensor(2);
        float *bo = bboxs_tensor.data<float>();

        //example
        float cx = bo[4] ;
        float cy = bo[5] ;
        float bx = bo[6] ;
        float by = bo[7] ;
        cv::rectangle(imageMat, cv::Rect(bo[0],bo[1], bo[2]-bo[0], bo[3]-bo[1]), cv::Scalar(0, 255, 0), 2);
        cv::rectangle(imageMat, cv::Rect(cx,cy, bx-cx, by-cy), cv::Scalar(0, 255, 0), 2);
        cv::imwrite("aimage.png",imageMat);
    }
    catch(const ov::Exception& e){
        std::cerr << e.what() << '\n';
    }
}
