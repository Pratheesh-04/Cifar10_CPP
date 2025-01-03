#include "utils.h"
#include <iostream>

std::vector<std::vector<std::vector<std::vector<float>>>> reshape_kernel(
    const std::vector<float>& kernels_flat, int num_kernels, int kernel_height, int kernel_width, int input_channels) {
    // Initialize the 4D kernel
    std::vector<std::vector<std::vector<std::vector<float>>>> kernel(
        num_kernels, std::vector<std::vector<std::vector<float>>>(
                         kernel_height, std::vector<std::vector<float>>(
                                            kernel_width, std::vector<float>(input_channels, 0))));

    int kernel_index = 0;
    for (int c = 0; c < num_kernels; ++c) {
        for (int h = 0; h < kernel_height; ++h) {
            for (int w = 0; w < kernel_width; ++w) {
                for (int ic = 0; ic < input_channels; ++ic) {
                    kernel[c][h][w][ic] = kernels_flat[kernel_index++];
                }
            }
        }
    }

    return kernel;
}

void display_output(const std::vector<std::vector<std::vector<float>>>& output, int channel) {
    int height = output.size();
    int width = output[0].size();
    std::cout << "Output for channel " << channel << ":\n";
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            std::cout << output[h][w][channel] << " ";
        }
        std::cout << std::endl;
    }
}

void display_flattened_output(const std::vector<float>& flattened_output) {
    std::cout << "Flattened Output: " << std::endl;
    for (size_t i = 0; i < flattened_output.size(); ++i) {
        if (i % 10 == 0) std::cout << std::endl;  // Print 10 elements per line
        std::cout << flattened_output[i] << " ";
    }
    std::cout << std::endl;
}

