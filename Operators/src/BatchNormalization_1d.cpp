#include "batchnormalization_1d.h"
#include <iostream>
#include <fstream> // For file handling
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <chrono> // For measuring execution time

void batch_normalization_1d(const std::vector<float>& input, std::vector<float>& output,
                             const std::vector<float>& gamma, const std::vector<float>& beta,
                             const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                             float epsilon, size_t channels, size_t height, size_t width, std::string layername) {
    // Start timer
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

    for (int i = 0; i < output.size(); ++i) {
        // ReLU: set negative values to 0
        output[i] = std::max(0.0f, output[i]);
    }

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    std::cout << "=====================================================\n";
    // Print execution time
    std::cout << "Execution time for batch_normalization_1d: " << execution_time.count() << " seconds\n";

    // Save the first channel to a text file
    std::ofstream outfile("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    // Debug output for the first channel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            // std::cout << std::fixed << std::setprecision(6) << output[idx] << " "; // Print to console
            outfile << output[idx] << " "; // Write to file
        }
        // std::cout << "\n";
        outfile << "\n";
    }
    outfile.close();
    std::cout << "Output of First channel saved to data/cpp_outputs/"+layername+".txt" << std::endl;
    std::cout << "=====================================================\n";
}

void batch_normalization_1d1(const std::vector<float>& input, std::vector<float>& output,
                             const std::vector<float>& gamma, const std::vector<float>& beta,
                             const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                             float epsilon, size_t channels, std::string layername) {
    // Start timer
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

    for (int i = 0; i < output.size(); ++i) {
        // ReLU: set negative values to 0
        output[i] = std::max(0.0f, output[i]);
    }

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    std::cout << "=====================================================\n";
    // Print execution time
    std::cout << "Execution time for batch_normalization_1d1: " << execution_time.count() << " seconds\n";

    // Save the output of all channels to a text file
    std::ofstream outfile("F:/MCW/c++ application/Project_Root/data/cpp_outputs/" + layername + ".txt");
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    for (size_t s = 0; s < spatial_size; ++s) {
        for (size_t c = 0; c < channels; ++c) {
            size_t idx = s * channels + c;
            outfile << output[idx] << " "; // Write all channel outputs for the current spatial position
        }
        outfile << std::endl; // New line after all channels for the current spatial position
    }

    outfile.close();
    std::cout << "Output of all channels saved to data/cpp_outputs/" + layername + ".txt" << std::endl;
    std::cout << "=====================================================\n";
}
