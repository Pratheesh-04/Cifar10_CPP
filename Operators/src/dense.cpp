#include "dense.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>

void softmax(std::vector<float>& tensor) {
    float max_val = *std::max_element(tensor.begin(), tensor.end());
    float sum = 0.0f;
    for (auto& value : tensor) {
        value = std::exp(value - max_val);
        sum += value;
    }
    for (auto& value : tensor) {
        value /= sum;
    }
}

void dense(const std::vector<float>& input, const std::vector<float>& weights,
           const std::vector<float>& bias, std::vector<float>& output,
           const std::array<int, 2>& input_shape, const std::array<int, 2>& output_shape,
           const std::string& activation) {

    // Validate the sizes of input, weights, and output
    int input_size = input_shape[1];
    int output_size = output_shape[1];

    std::cout << "Input size: " << input.size() << std::endl;
    std::cout << "Weights size: " << weights.size() << std::endl;
    std::cout << "Bias size: " << bias.size() << std::endl;

    output.resize(output_size, 0.0f); 

    if (weights.size() != input_size * output_size) {
        throw std::runtime_error("Weights size does not match input_size * output_size.");
    }

    if (bias.size() != output_size) {
        throw std::runtime_error("Bias size does not match output_size.");
    }

    // Perform the dense operation (matrix multiplication + bias)
    for (int o = 0; o < output_size; ++o) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i * output_size + o];
        }
        output[o] = sum + bias[o];
    }

    // Apply activation if needed
    if (activation == "softmax") {
        softmax(output);
    }
    std::cout << "Output size: " << output.size() << std::endl;
    // Print output for debugging
    for (int i = 0 ; i < 10 && i < output_size; i++) {
        std::cout << output[i] << " " << std::endl;
    }

    std::cout << "Dense Output Shape: [" << output_size << "]" << std::endl;
}
