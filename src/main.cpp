    #include "../Operators/include/maxpooling.h"
    #include "../Operators/include/conv2d.h"
    #include "../utilities/include/utils.h"
    #include <nlohmann/json.hpp>
    #include <fstream>
    #include <iostream>
    #include <vector>
    #include <cassert>
    #include "dense.h"
    #include "batchnormalization_1d.h"

    using json = nlohmann::json;
    using namespace std;

    int main() {
        // Load JSON configuration
        ifstream json_file("F:/MCW/c++ application/Project_Root/configs/config_file.json");
        assert(json_file.is_open() && "Unable to open JSON configuration file");

        json config;
        json_file >> config;
        json_file.close();

        // Access layers array from config
        auto layers = config["layers"];

        // Variables to store Conv2D attributes
        string base_path = "F:/MCW/c++ application/Project_Root/";
        string output_file = "F:/MCW/c++ application/Project_Root/layer_outputs.tx/";

        // Initialize a shared variable for layer outputs
        vector<float> layer_output;
        vector<int> strides;
        string padding;

        // Process each layer sequentially
        for (const auto& layer : layers) {
            if (layer["layer_name"] == "conv2d") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 2 && "Invalid number of weight paths in the configuration file for conv2d layer");

                string kernel_path = base_path + weights_file_paths[0];
                string bias_path = base_path + weights_file_paths[1];

                int input_height = layer["attributes"]["input_shape"][1];
                int input_width = layer["attributes"]["input_shape"][2];
                int input_channels = layer["attributes"]["input_shape"][3];
                int stride_height = layer["attributes"]["strides"][0];
                int stride_width = layer["attributes"]["strides"][1];
                
                // Initialize input (if it's the first layer)
                if (layer_output.empty()) {
                    layer_output = vector<float>(input_height * input_width * input_channels, 1.0);
                }

                int kernel_height = layer["attributes"]["kernel_size"][0];
                int kernel_width = layer["attributes"]["kernel_size"][0];
                int output_channels = layer["attributes"]["output_shape"][3];
                padding = layer["attributes"]["padding"];
            
                vector<float> kernels_flat = load_binary_data(kernel_path, kernel_height * kernel_width * output_channels * input_channels);
                vector<float> biases = load_binary_data(bias_path, output_channels);

                
                // Perform Conv2D operation
                vector<float> conv2d_output;
                conv2d_1d(layer_output, kernels_flat, biases, conv2d_output, 
                input_height, input_width, input_channels,
                kernel_height, kernel_width, output_channels,
                stride_height, padding, layer["layer_name"]);

                // Update the shared layer output
                layer_output = conv2d_output;
                // write_to_file(output_file, layer_output);

                // Display output for the first channel (channel 0) after Conv2D
                cout << "===============================================================================" << endl;
            }

            if (layer["layer_name"] == "batch_normalization") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 4 && "Invalid number of weight paths in the configuration file for batch_normalization layer");

                string gamma_path = base_path + weights_file_paths[0];
                string beta_path = base_path + weights_file_paths[1];
                string moving_mean_path = base_path + weights_file_paths[2];
                string moving_variance_path = base_path + weights_file_paths[3];

                int output_channels = layer["attributes"]["output_shape"][3];

                vector<float> gamma = load_binary_data(gamma_path, output_channels);
                vector<float> beta = load_binary_data(beta_path, output_channels);
                vector<float> moving_mean = load_binary_data(moving_mean_path, output_channels);
                vector<float> moving_variance = load_binary_data(moving_variance_path, output_channels);
                float epsilon = 1e-5;
                int input_height = layer["attributes"]["input_shape"][1];
                int input_width = layer["attributes"]["input_shape"][2]; 

                // Perform Batch Normalization
                vector<float> batch_normalized_output;
                batch_normalization_1d(layer_output, batch_normalized_output, gamma, beta, moving_mean, moving_variance, epsilon, output_channels,input_height, input_width, layer["layer_name"]);

                // Update the shared layer output
                layer_output = batch_normalized_output;

            }

            if (layer["layer_name"] == "max_pooling2d") {
                std::array<int, 2> strides = {layer["attributes"]["strides"][0], layer["attributes"]["strides"][1]};
                string padding = layer["attributes"]["padding"].get<string>();
                std::array<int, 4> input_shape = { 1, layer["attributes"]["input_shape"][1], layer["attributes"]["input_shape"][2], layer["attributes"]["input_shape"][3]};
                std::array<int, 4> output_shape = { 1, layer["attributes"]["output_shape"][1], layer["attributes"]["output_shape"][2], layer["attributes"]["output_shape"][3] };
                std::array<int, 2> pool_size = {2,2};
                
                vector<float> maxpooling_output;
                max_pooling2d(layer_output,maxpooling_output,input_shape,output_shape,pool_size,strides,padding, layer["layer_name"]);
                
                layer_output = maxpooling_output;
            }

            if (layer["layer_name"] == "conv2d_1") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 2 && "Invalid number of weight paths in the configuration file for conv2d layer");

                string kernel_path = base_path + weights_file_paths[0];
                string bias_path = base_path + weights_file_paths[1];

                int input_height = layer["attributes"]["input_shape"][1];
                int input_width = layer["attributes"]["input_shape"][2];
                int input_channels = layer["attributes"]["input_shape"][3];
                int stride_height = layer["attributes"]["strides"][0];
                int stride_width = layer["attributes"]["strides"][1];
                int kernel_height = layer["attributes"]["kernel_size"][0];
                int kernel_width = layer["attributes"]["kernel_size"][0];
                int output_channels = layer["attributes"]["output_shape"][3];
                padding = layer["attributes"]["padding"];
            
                vector<float> kernels_flat = load_binary_data(kernel_path, kernel_height * kernel_width * output_channels * input_channels);
                vector<float> biases = load_binary_data(bias_path, output_channels);

                
                // Perform Conv2D operation
                vector<float> conv2d_output;
                conv2d_1d(layer_output, kernels_flat, biases, conv2d_output, 
                input_height, input_width, input_channels,
                kernel_height, kernel_width, output_channels,
                stride_height, padding, layer["layer_name"]);

                // Update the shared layer output
                layer_output = conv2d_output;
            }
            if (layer["layer_name"] == "batch_normalization_1") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 4 && "Invalid number of weight paths in the configuration file for batch_normalization layer");

                string gamma_path = base_path + weights_file_paths[0];
                string beta_path = base_path + weights_file_paths[1];
                string moving_mean_path = base_path + weights_file_paths[2];
                string moving_variance_path = base_path + weights_file_paths[3];

                int output_channels = layer["attributes"]["output_shape"][3];

                vector<float> gamma = load_binary_data(gamma_path, output_channels);
                vector<float> beta = load_binary_data(beta_path, output_channels);
                vector<float> moving_mean = load_binary_data(moving_mean_path, output_channels);
                vector<float> moving_variance = load_binary_data(moving_variance_path, output_channels);
                float epsilon = 1e-5;
                int input_height = layer["attributes"]["input_shape"][1];
                int input_width = layer["attributes"]["input_shape"][2]; 

                // Perform Batch Normalization
                vector<float> batch_normalized_output;
                batch_normalization_1d(layer_output, batch_normalized_output, gamma, beta, moving_mean, moving_variance, epsilon, output_channels,input_height, input_width, layer["layer_name"]);

                // Update the shared layer output
                layer_output = batch_normalized_output;
            }

            if (layer["layer_name"] == "max_pooling2d_1") {
                std::array<int, 2> strides = {layer["attributes"]["strides"][0], layer["attributes"]["strides"][1]};
                string padding = layer["attributes"]["padding"].get<string>();
                std::array<int, 4> input_shape = { 1, layer["attributes"]["input_shape"][1], layer["attributes"]["input_shape"][2], layer["attributes"]["input_shape"][3]};
                std::array<int, 4> output_shape = { 1, layer["attributes"]["output_shape"][1], layer["attributes"]["output_shape"][2], layer["attributes"]["output_shape"][3] };
                std::array<int, 2> pool_size = {2,2};
                
                vector<float> maxpooling_output;
                max_pooling2d(layer_output,maxpooling_output,input_shape,output_shape,pool_size,strides,padding, layer["layer_name"]);
                
                layer_output = maxpooling_output;
            }

            if (layer["layer_name"] == "conv2d_2") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 2 && "Invalid number of weight paths in the configuration file for conv2d layer");

                string kernel_path = base_path + weights_file_paths[0];
                string bias_path = base_path + weights_file_paths[1];

                int input_height = layer["attributes"]["input_shape"][1];
                int input_width = layer["attributes"]["input_shape"][2];
                int input_channels = layer["attributes"]["input_shape"][3];
                int stride_height = layer["attributes"]["strides"][0];
                int stride_width = layer["attributes"]["strides"][1];
                int kernel_height = layer["attributes"]["kernel_size"][0];
                int kernel_width = layer["attributes"]["kernel_size"][0];
                int output_channels = layer["attributes"]["output_shape"][3];
                padding = layer["attributes"]["padding"];
            
                vector<float> kernels_flat = load_binary_data(kernel_path, kernel_height * kernel_width * output_channels * input_channels);
                vector<float> biases = load_binary_data(bias_path, output_channels);

                
                // Perform Conv2D operation
                vector<float> conv2d_output;
                conv2d_1d(layer_output, kernels_flat, biases, conv2d_output, 
                input_height, input_width, input_channels,
                kernel_height, kernel_width, output_channels,
                stride_height, padding, layer["layer_name"]);

                // Update the shared layer output
                layer_output = conv2d_output;
            }
            if (layer["layer_name"] == "batch_normalization_2") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 4 && "Invalid number of weight paths in the configuration file for batch_normalization layer");

                string gamma_path = base_path + weights_file_paths[0];
                string beta_path = base_path + weights_file_paths[1];
                string moving_mean_path = base_path + weights_file_paths[2];
                string moving_variance_path = base_path + weights_file_paths[3];

                int output_channels = layer["attributes"]["output_shape"][3];

                vector<float> gamma = load_binary_data(gamma_path, output_channels);
                vector<float> beta = load_binary_data(beta_path, output_channels);
                vector<float> moving_mean = load_binary_data(moving_mean_path, output_channels);
                vector<float> moving_variance = load_binary_data(moving_variance_path, output_channels);
                float epsilon = 1e-5;
                int input_height = layer["attributes"]["input_shape"][1];
                int input_width = layer["attributes"]["input_shape"][2]; 

                // Perform Batch Normalization
                vector<float> batch_normalized_output;
                batch_normalization_1d(layer_output, batch_normalized_output, gamma, beta, moving_mean, moving_variance, epsilon, output_channels,input_height, input_width, layer["layer_name"]);

                // Update the shared layer output
                layer_output = batch_normalized_output;
            }

            if (layer["layer_name"] == "dense") {
    // Load the weights and biases for the Dense layer
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 2 && "Invalid number of weight paths in the configuration file for dense layer");

                string weights_path = base_path + weights_file_paths[0];
                string bias_path = base_path + weights_file_paths[1];

                // Load weights and biases
                vector<float> weights = load_binary_data(weights_path, layer_output.size() * layer["attributes"]["output_shape"][1]);
                vector<float> biases = load_binary_data(bias_path, layer["attributes"]["output_shape"][1]);

                std::array<int, 2> dense_input_shape = {1,layer["attributes"]["input_shape"][1]};
                std::array<int, 2> dense_output_shape = {1,layer["attributes"]["output_shape"][1]};
                std::string activation = layer["attributes"]["activation"];
                // Create a Dense layer instance
                vector<float> dense_output;
                dense(layer_output,weights,biases, dense_output,dense_input_shape,dense_output_shape,activation);
                layer_output = dense_output;
            }

            if (layer["layer_name"] == "batch_normalization_3") {
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 4 && "Invalid number of weight paths in the configuration file for batch_normalization layer");

                string gamma_path = base_path + weights_file_paths[0];
                string beta_path = base_path + weights_file_paths[1];
                string moving_mean_path = base_path + weights_file_paths[2];
                string moving_variance_path = base_path + weights_file_paths[3];

                int output_channels = layer["attributes"]["output_shape"][1];

                vector<float> gamma = load_binary_data(gamma_path, output_channels);
                vector<float> beta = load_binary_data(beta_path, output_channels);
                vector<float> moving_mean = load_binary_data(moving_mean_path, output_channels);
                vector<float> moving_variance = load_binary_data(moving_variance_path, output_channels);
                float epsilon = 1e-5;
                int input_height = 1;
                int input_width = layer["attributes"]["input_shape"][1]; 

                // Perform Batch Normalization
                vector<float> batch_normalized_output;
                batch_normalization_1d1(layer_output, batch_normalized_output, gamma, beta, moving_mean, moving_variance, epsilon, output_channels, layer["layer_name"]);
                 // Update the shared layer output
                layer_output = batch_normalized_output;
            }

            if (layer["layer_name"] == "dense_1") {
    // Load the weights and biases for the Dense layer
                vector<string> weights_file_paths = layer["weights_file_paths"].get<vector<string>>();
                assert(weights_file_paths.size() == 2 && "Invalid number of weight paths in the configuration file for dense layer");

                string weights_path = base_path + weights_file_paths[0];
                string bias_path = base_path + weights_file_paths[1];

                // Load weights and biases
                vector<float> weights = load_binary_data(weights_path, layer_output.size() * layer["attributes"]["output_shape"][1]);
                vector<float> biases = load_binary_data(bias_path, layer["attributes"]["output_shape"][1]);

                std::array<int, 2> dense_input_shape = {1,layer["attributes"]["input_shape"][1]};
                std::array<int, 2> dense_output_shape = {1,layer["attributes"]["output_shape"][1]};
                std::string activation = layer["attributes"]["activation"];
                // Create a Dense layer instance
                vector<float> dense_output;
                dense(layer_output,weights,biases, dense_output,dense_input_shape,dense_output_shape,activation);
                layer_output = dense_output;
            }
        }
        return 0;
    }
