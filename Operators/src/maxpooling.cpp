#include "maxpooling.h"
#include <iostream>
#include <cmath>
#include <limits>

void max_pooling2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
                   const std::array<int, 2>& pool_size, const std::array<int, 2>& strides,
                   const std::string& padding) {
    int batch = input_shape[0];
    int in_height = input_shape[1], in_width = input_shape[2], in_channels = input_shape[3];
    int out_height = (in_height - pool_size[0]) / strides[0] + 1;
    int out_width = (in_width - pool_size[1]) / strides[1] + 1;


    output.resize(batch * out_height * out_width * in_channels);


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
                                
                                // Debugging: Check if input_idx is within bounds
                                if (input_idx >= input.size() || input_idx < 0) {
                                    std::cout << "Input index out of range! index: " << input_idx << std::endl;
                                }
                                
                                max_val = std::max(max_val, input[input_idx]);
                            }
                        }
                    }

                    int output_idx = ((b * out_height + h) * out_width + w) * in_channels + c;
                    
                    // Debugging: Check if output_idx is within bounds
                    if (output_idx >= output.size() || output_idx < 0) {
                        std::cout << "Output index out of range! index: " << output_idx << std::endl;
                    }

                    output[output_idx] = max_val;
                }
            }
        }
    }

    std::cout << "First Channel Output (as Matrix) for Batch 0:" << std::endl;
    for (int h = 0; h < out_height; ++h) {
        for (int w = 0; w < out_width; ++w) {
            int output_idx = ((0 * out_height + h) * out_width + w) * in_channels;
            std::cout << output[output_idx] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "MaxPooling2D Output Shape: [" << batch << ", " << out_height << ", " << out_width << ", " << input_shape[3] << "]" << std::endl;
    std::cout << "====================================================";
}
