#include "maxpooling.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <fstream>   // For file handling
#include <chrono>    // For timing

void max_pooling2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
                   const std::array<int, 2>& pool_size, const std::array<int, 2>& strides,
                   const std::string& padding, const std::string& layername) {
    int batch = input_shape[0];
    int in_height = input_shape[1], in_width = input_shape[2], in_channels = input_shape[3];
    int out_height = (in_height - pool_size[0]) / strides[0] + 1;
    int out_width = (in_width - pool_size[1]) / strides[1] + 1;

    output.resize(batch * out_height * out_width * in_channels);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Open the output file
    std::ofstream output_file("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open file " << std::endl;
        return;
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                for (int c = 0; c < in_channels; ++c) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int ph = 0; ph < pool_size[0]; ++ph) {
                        for (int pw = 0; pw < pool_size[1]; ++pw) {
                            int ih = h * strides[0] + ph;
                            int iw = w * strides[1] + pw;

                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + c;

                                max_val = std::max(max_val, input[input_idx]);
                            }
                        }
                    }

                    int output_idx = ((b * out_height + h) * out_width + w) * in_channels + c;
                    output[output_idx] = max_val;

                    // Write result to file
                    output_file << max_val << " ";
                }
                output_file << "\n"; // Newline after each channel
            }
        }
    }

    // Close the file
    output_file.close();

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Print execution time
    std::cout << "Execution Time: " << elapsed_time.count() << " seconds" << std::endl;

    std::cout << "Results written to file: " << "F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt" << std::endl;
    std::cout << "MaxPooling2D Output Shape: [" << batch << ", " << out_height << ", " << out_width << ", " << input_shape[3] << "]" << std::endl;
    std::cout << "====================================================" << std::endl;
}
