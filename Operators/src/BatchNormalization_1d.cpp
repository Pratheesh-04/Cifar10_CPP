#include "batchnormalization_1d.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <chrono> // For measuring execution time

void batch_normalization_1d(const std::vector<float>& input, std::vector<float>& output,
                             const std::vector<float>& gamma, const std::vector<float>& beta,
                             const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                             float epsilon, size_t channels, size_t height, size_t width,
                             std::string layername ) {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate the spatial size per channel
    size_t spatial_size = height * width;

    // Ensure output is resized correctly
    output.resize(input.size());

    // Perform batch normalization
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Calculate index for the flattened 1D array
                int idx = (h * width + w) * channels + c;
                output[idx] = gamma[c] * (input[idx] - moving_mean[c]) /
                              std::sqrt(moving_variance[c] + epsilon) + beta[c];
            }
        }
    }

    // Apply ReLU activation
    for (int i = 0; i < output.size(); ++i) {
        output[i] = std::max(0.0f, output[i]);
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // Write the output to the specified file
    std::ofstream file("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open file ");
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            file << std::fixed << std::setprecision(6) << output[idx] << std::endl;
        }
    }
    file.close();

    // Print output shape
    std::cout << "=====================================================" << std::endl;
    std::cout << "Output Shape: [" << height << ", " << width << ", " << channels << "]" << std::endl;
    std::cout << "Execution Time (microseconds): " << execution_time << "\n";
    std::cout << "Layer output successfully written to " << "F:/MCW/c++ application/Project_Root/data/cpp_outputs/" + layername + ".txt" << std::endl;
    std::cout << "=====================================================" << std::endl;
}

void batch_normalization_1d1(const std::vector<float>& input, std::vector<float>& output,
                            const std::vector<float>& gamma, const std::vector<float>& beta,
                            const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                            float epsilon, size_t channels, std::string layername) {
    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    if (input.size() % channels != 0) {
        throw std::runtime_error("Input size is not divisible by the number of channels.");
    }

    size_t spatial_size = input.size() / channels;
    output.resize(input.size());

    for (size_t c = 0; c < channels; ++c) {
        for (size_t s = 0; s < spatial_size; ++s) {
            size_t idx = s * channels + c;
            output[idx] = gamma[c] * (input[idx] - moving_mean[c]) /
                          std::sqrt(moving_variance[c] + epsilon) + beta[c];
        }
    }

    // Apply ReLU activation
    for (int i = 0; i < output.size(); ++i) {
        output[i] = std::max(0.0f, output[i]);
    }

    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // Write the output to the specified file
    std::ofstream outputFile("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");

    if (!outputFile.is_open()) {
        throw std::runtime_error("Error: Unable to open file ");
    }

    for (size_t i = 0; i < output.size(); ++i) {
        outputFile << std::fixed << std::setprecision(6) << output[i] << std::endl;
    }
    outputFile.close();

    // Print output shape
    std::cout << "=====================================================" << std::endl;
    std::cout << "Batch Normalization Output Shape: [" << spatial_size << ", " << channels << "]" << std::endl;
    std::cout << "Execution Time (microseconds): " << execution_time << "\n";
    std::cout << "Layer output successfully written to " << "F:/MCW/c++ application/Project_Root/data/cpp_outputs/" + layername + ".txt" << std::endl;
    std::cout << "=====================================================" << std::endl;
}
