#include "batchnormalization_1d.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <stdexcept>

void batch_normalization_1d(const std::vector<float>& input, std::vector<float>& output,
                             const std::vector<float>& gamma, const std::vector<float>& beta,
                             const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                             float epsilon, size_t channels, size_t height, size_t width) {
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

    // Print debug information for the first channel (optional)
    std::cout << "Batch Normalization" << std::endl;
    std::cout << "First channel of output:\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            std::cout << std::fixed << std::setprecision(6) << output[idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "====================\n";

    for (int i = 0; i < output.size(); ++i) {
        // ReLU: set negative values to 0
        output[i] = std::max(0.0f, output[i]);
    }
}

void batch_normalization_1d1(const std::vector<float>& input, std::vector<float>& output,
                            const std::vector<float>& gamma, const std::vector<float>& beta,
                            const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                            float epsilon, size_t channels) {
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

    std::cout << "Batch Normalization Output Values:" << std::endl;
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << output[i] << " ";
            if ((i + 1) % 10 == 0) {
                std::cout << std::endl; // Print 10 values per line
            }
        }
    std::cout << std::endl;
    std::cout << "Batch normalization completed. Output size: " << output.size() << std::endl;

    for (int i = 0; i < output.size(); ++i) {
        // ReLU: set negative values to 0
        output[i] = std::max(0.0f, output[i]);
    }
}