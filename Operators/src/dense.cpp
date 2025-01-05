#include "dense.h"
#include <iostream>
#include <fstream> // For file handling
#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <chrono> // For timing

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
           const std::string& activation, std::string layername) {

    // Validate the sizes of input, weights, and output
    int input_size = input_shape[1];
    int output_size = output_shape[1];

    output.resize(output_size, 0.0f); 

    if (weights.size() != input_size * output_size) {
        throw std::runtime_error("Weights size does not match input_size * output_size.");
    }

    if (bias.size() != output_size) {
        throw std::runtime_error("Bias size does not match output_size.");
    }

    // Start timing the dense operation
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform the dense operation (matrix multiplication + bias)
    for (int o = 0; o < output_size; ++o) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i * output_size + o];
        }
        output[o] = sum + bias[o];
    }

    // End timing the dense operation
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    // Apply activation if needed
    if (activation == "softmax") {
        softmax(output);
    }

    // Print dense layer shape
    std::cout << "Dense Output Shape: [" << output_size << "]" << std::endl;

    // Print execution time
    std::cout << "Dense Layer Execution Time: " << execution_time.count() << " seconds" << std::endl;

    // Write the output to a text file
    std::ofstream outfile("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");
    if (outfile.is_open()) {
        for (int i = 0; i < output_size; i++) {
            outfile << output[i] << " ";
        }
        outfile << "\n";
        outfile.close();
    std::cout << "Output of First channel saved to data/cpp_outputs/"+layername+".txt" << std::endl;
    } else {
        std::cerr << "Error opening file for writing!" << std::endl;
    }

    std::cout << "=====================================================\n";
}
