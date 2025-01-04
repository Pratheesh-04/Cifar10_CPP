#include "dense.h"
#include <iostream>
#include <fstream>  // For file handling
#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <chrono>  // For timing

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
           const std::string& activation, const std::string& layername) {
    int input_size = input_shape[1];
    int output_size = output_shape[1];

    output.resize(output_size, 0.0f);

    if (weights.size() != input_size * output_size) {
        throw std::runtime_error("Weights size does not match input_size * output_size.");
    }

    if (bias.size() != output_size) {
        throw std::runtime_error("Bias size does not match output_size.");
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

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

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Save results to file
    std::ofstream output_file("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open file " << std::endl;
        return;
    }

    for (const auto& value : output) {
        output_file << value << "\n";
    }
    output_file.close();

    // Print results
    std::cout<< "=====================================================" << std::endl;
    std::cout << "Dense Output Shape: [" << output_size << "]" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " seconds" << std::endl;
    std::cout << "Results written to file: " << "F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt" << std::endl;
    std::cout<< "=====================================================";
}
