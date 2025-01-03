#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

std::vector<std::vector<std::vector<std::vector<float>>>> reshape_kernel(
    const std::vector<float>& kernels_flat, int num_kernels, int kernel_height, int kernel_width, int input_channels);

// Function to display output for a specific channel
void display_output(const std::vector<std::vector<std::vector<float>>>& output, int channel);

void display_flattened_output(const std::vector<float>& flattened_output);

#endif // UTILS_H
