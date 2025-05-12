
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <array>


std::vector<const char*> input_names = { "images", "orig_target_sizes" };
std::vector<const char*> output_names = { "labels", "boxes", "scores" };

/**
 * @brief Draws bounding boxes, labels, and confidence scores on an image.
 *
 * This function takes an image, a list of labels, bounding boxes, and their corresponding confidence scores,
 * and overlays the bounding boxes and labels on the image. The bounding boxes are adjusted to compensate
 * for resizing and padding applied during preprocessing.
 *
 * @param image The input image (cv::Mat) on which to draw the bounding boxes and labels.
 * @param labels A vector of integer labels corresponding to detected objects.
 * @param boxes A vector of bounding boxes, where each box is represented as {x1, y1, x2, y2}.
 * @param scores A vector of confidence scores corresponding to the bounding boxes.
 * @param ratio The scaling factor used to resize the image during preprocessing.
 * @param pad_w The horizontal padding applied to the image during preprocessing.
 * @param pad_h The vertical padding applied to the image during preprocessing.
 * @param thrh The confidence threshold; only boxes with scores above this value will be drawn (default is 0.4).
 * @return A cv::Mat object containing the original image with bounding boxes, labels, and scores drawn on it.
 */
cv::Mat draw(
	const cv::Mat& image,
	const std::vector<int64_t>& labels,
	const std::vector<std::vector<float>>& boxes,
	const std::vector<float>& scores,
	float ratio,
	int pad_w,
	int pad_h,
	float thrh = 0.4)
{
	// Clone the input image to preserve the original image
	cv::Mat img = image.clone();

	// Iterate over all detected objects
	for (size_t i = 0; i < scores.size(); ++i) {
		// Only process objects with confidence scores above the threshold
		if (scores[i] > thrh) {
			// Adjust bounding box coordinates to account for resizing and padding
			float x1 = (boxes[i][0] - pad_w) / ratio; // Top-left x-coordinate
			float y1 = (boxes[i][1] - pad_h) / ratio; // Top-left y-coordinate
			float x2 = (boxes[i][2] - pad_w) / ratio; // Bottom-right x-coordinate
			float y2 = (boxes[i][3] - pad_h) / ratio; // Bottom-right y-coordinate

			// Draw the bounding box on the image
			cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);

			// Prepare the label text with class label and confidence score
			std::string label_text = "Label: " + std::to_string(labels[i]) +
				" Conf: " + std::to_string(scores[i]);

			// Draw the label text above the bounding box
			cv::putText(img, label_text, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
		}
	}

	// Return the annotated image
	return img;
}



/**
 * @brief Resizes an image while maintaining its aspect ratio and pads the resized image to a square of a specified size.
 *
 * This function scales the input image proportionally to fit within a square of the specified size while preserving
 * the aspect ratio. It then pads the resized image with black pixels (value 0) to fill the remaining space, creating
 * a square output image.
 *
 * @param image Input image (cv::Mat) to be resized and padded.
 * @param size Target size of the square output image (both width and height will be equal to size).
 * @param ratio Output parameter that will contain the scaling factor applied to the image.
 * @param pad_w Output parameter that will contain the width of padding applied on the left and right sides.
 * @param pad_h Output parameter that will contain the height of padding applied on the top and bottom sides.
 * @return A cv::Mat object containing the resized and padded square image.
 */
cv::Mat resizeWithAspectRatio(const cv::Mat& image, int size, float& ratio, int& pad_w, int& pad_h) {
	// Get the original width and height of the input image
	int original_width = image.cols;
	int original_height = image.rows;

	// Compute the scaling ratio to fit the image within the target size while maintaining aspect ratio
	ratio = std::min(static_cast<float>(size) / original_width, static_cast<float>(size) / original_height);
	int new_width = static_cast<int>(original_width * ratio);  // New width after scaling
	int new_height = static_cast<int>(original_height * ratio); // New height after scaling

	// Resize the image using the computed dimensions
	cv::Mat resized_image;
	cv::resize(image, resized_image, cv::Size(new_width, new_height));

	// Calculate the padding required to center the resized image in the square output
	pad_w = (size - new_width) / 2; // Horizontal padding (left and right)
	pad_h = (size - new_height) / 2; // Vertical padding (top and bottom)

	// Create a square output image filled with black pixels (value 0)
	cv::Mat padded_image(size, size, resized_image.type(), cv::Scalar(0, 0, 0));

	// Copy the resized image into the center of the square output image
	resized_image.copyTo(padded_image(cv::Rect(pad_w, pad_h, new_width, new_height)));

	// Return the resized and padded image
	return padded_image;
}

/**
 * @brief Preprocess an input image, run inference using an ONNX model, and process the results.
 *
 * This function resizes the input image while maintaining its aspect ratio, prepares it for inference,
 * runs the inference using the specified ONNX Runtime session, and processes the output to draw
 * bounding boxes and labels on the original image.
 *
 * @param session The ONNX Runtime session used to perform inference.
 * @param image The input image (OpenCV Mat) to process.
 * @return cv::Mat The result image with bounding boxes and labels drawn.
 */
cv::Mat processImage(Ort::Session& session, const cv::Mat& image) {
	float ratio;         // Aspect ratio for resizing the image.
	int pad_w, pad_h;    // Padding added to maintain aspect ratio.
	int target_size = 640; // Target size for resizing (typically square).

	// Step 1: Resize and pad the image to the target size while preserving the aspect ratio.
	cv::Mat resized_image = resizeWithAspectRatio(image, target_size, ratio, pad_w, pad_h);

	// Step 2: Convert the resized image to RGB format as required by the model.
	cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

	// Step 3: Prepare the input tensor in NCHW format (channels-first).
	std::vector<int64_t> input_dims = { 1, 3, target_size, target_size }; // Batch size = 1, Channels = 3, HxW = target_size.
	std::vector<float> input_tensor_values(input_dims[1] * input_dims[2] * input_dims[3]);

	// Populate the input tensor with normalized pixel values (range 0 to 1).
	int index = 0;
	for (int c = 0; c < 3; ++c) { // Loop through channels.
		for (int i = 0; i < resized_image.rows; ++i) { // Loop through rows.
			for (int j = 0; j < resized_image.cols; ++j) { // Loop through columns.
				input_tensor_values[index++] = resized_image.at<cv::Vec3b>(i, j)[c] / 255.0f; // Normalize pixel value.
			}
		}
	}

	// Step 4: Create ONNX Runtime input tensors.
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

	// Tensor for the preprocessed image.
	Ort::Value input_tensor_images = Ort::Value::CreateTensor<float>(
		memory_info, input_tensor_values.data(), input_tensor_values.size(),
		input_dims.data(), input_dims.size()
	);

	// Tensor for the original target sizes (optional, used for postprocessing).
	std::vector<int64_t> orig_size_dims = { 1, 2 };
	std::vector<int64_t> orig_size_values = {
		static_cast<int64_t>(resized_image.rows),
		static_cast<int64_t>(resized_image.cols)
	};
	Ort::Value input_tensor_orig_target_sizes = Ort::Value::CreateTensor<int64_t>(
		memory_info, orig_size_values.data(), orig_size_values.size(),
		orig_size_dims.data(), orig_size_dims.size()
	);

	// Step 5: Run inference on the session.
	auto outputs = session.Run(
		Ort::RunOptions{ nullptr },                            // Default run options.
		input_names.data(),                                    // Names of input nodes.
		std::array<Ort::Value, 2>{std::move(input_tensor_images), std::move(input_tensor_orig_target_sizes)}.data(),
		input_names.size(),                                    // Number of inputs.
		output_names.data(),                                   // Names of output nodes.
		output_names.size()                                    // Number of outputs.
	);

	// Step 6: Extract and process model outputs.
	auto labels_ptr = outputs[0].GetTensorMutableData<int64_t>();  // Labels for detected objects.
	auto boxes_ptr = outputs[1].GetTensorMutableData<float>();     // Bounding boxes.
	auto scores_ptr = outputs[2].GetTensorMutableData<float>();    // Confidence scores.

	size_t num_boxes = outputs[2].GetTensorTypeAndShapeInfo().GetShape()[1]; // Number of detected boxes.

	// Convert raw output to structured data.
	std::vector<int64_t> labels(labels_ptr, labels_ptr + num_boxes);
	std::vector<std::vector<float>> boxes;
	std::vector<float> scores(scores_ptr, scores_ptr + num_boxes);

	auto boxes_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	size_t num_coordinates = boxes_shape[2]; // Usually 4 coordinates: (x1, y1, x2, y2).

	// Populate the `boxes` vector.
	for (size_t i = 0; i < num_boxes; ++i) {
		boxes.push_back({
			boxes_ptr[i * num_coordinates + 0], // x1
			boxes_ptr[i * num_coordinates + 1], // y1
			boxes_ptr[i * num_coordinates + 2], // x2
			boxes_ptr[i * num_coordinates + 3]  // y2
			});
	}

	// Step 7: Draw the results on the original image.
	cv::Mat result_image = draw(image, labels, boxes, scores, ratio, pad_w, pad_h);

	// Return the annotated image.
	return result_image;
}

/**
 * @brief Entry point of the application to perform object detection on an input source using a specified model.
 *
 * The program loads a pre-trained model, processes an input source (image, video, or webcam), and performs object
 * detection using either a CPU or GPU for computation. The results are displayed or saved as appropriate.
 *
 * @param argc The number of command-line arguments passed to the program.
 * @param argv The array of command-line arguments:
 *             - argv[0]: The name of the executable.
 *             - argv[1]: The path to the pre-trained model file.
 *             - argv[2]: The source of the input (image file, video file, or webcam index).
 *             - argv[3]: Flag to indicate whether to use GPU (1 for GPU, 0 for CPU).
 * @return Exit status:
 *         - Returns 0 on success.
 *         - Returns -1 if incorrect arguments are provided.
 */
int main(int argc, char** argv) {
	// Check if the required number of arguments is provided
	if (argc < 4) {
		// Display usage instructions if arguments are insufficient
		std::cerr << "Usage: " << argv[0]
			<< " <modelPath> <source[imagePath|videoPath|webcam]> <useGPU[1/0]>\n";
		return -1;
	}

	// Parse arguments
	std::string modelPath = argv[1];
	std::string source = argv[2];
	bool useGPU = std::stoi(argv[3]) != 0;

	// Initialize ONNX Runtime environment
	Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ONNXExample");
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	if (useGPU) {
		OrtCUDAProviderOptions cudaOptions;
		cudaOptions.device_id = 0; // Default to GPU 0
		session_options.AppendExecutionProvider_CUDA(cudaOptions);
		std::cout << "Using GPU for inference.\n";
	}
	else {
		std::cout << "Using CPU for inference.\n";
	}

	// Load ONNX model
	std::wstring widestr = std::wstring(modelPath.begin(), modelPath.end());
	const wchar_t* model_path = widestr.c_str();
	Ort::Session session(env, model_path, session_options);

	// Open source
	cv::VideoCapture cap;
	bool isVideo = false;
	bool isWebcam = false;
	bool isImage = false;
	cv::Mat frame;

	if (source == "webcam") {
		isWebcam = true;
		cap.open(0); // Open webcam
	}
	else if (source.find(".mp4") != std::string::npos ||
		source.find(".avi") != std::string::npos ||
		source.find(".mkv") != std::string::npos) {
		isVideo = true;
		cap.open(source); // Open video file
	}
	else {
		isImage = true;
		frame = cv::imread(source);
		if (frame.empty()) {
			std::cerr << "Error: Could not read image file.\n";
			return -1;
		}
	}

	if ((isVideo || isWebcam) && !cap.isOpened()) {
		std::cerr << "Error: Could not open video source.\n";
		return -1;
	}

	// Process source
	do {
		if (isWebcam || isVideo) {
			cap >> frame;
			if (frame.empty()) {
				if (isVideo) {
					std::cout << "End of video reached.\n";
				}
				break;
			}
		}

		// Process the frame/image with ONNX model
		auto result_image = processImage(session, frame);

		cv::imshow("ONNX Result", result_image);
		if (isImage) {
			cv::waitKey(0); // Wait indefinitely for image
			break;
		}
		else if (cv::waitKey(1) == 27) { // Exit on 'Esc' key for video/webcam
			break;
		}

		// FPS calculation for video/webcam
		static int frame_count = 0;
		static auto last_time = std::chrono::high_resolution_clock::now();
		frame_count++;
		auto current_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = current_time - last_time;
		if (elapsed.count() >= 1.0) {
			std::cout << "FPS: " << frame_count / elapsed.count() << "\n";
			frame_count = 0;
			last_time = current_time;
		}

	} while (isWebcam || isVideo);

	return 0;
}
