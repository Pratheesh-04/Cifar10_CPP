#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <string>

using namespace std;

/**
 * @brief Performs a 2D convolution on a flattened input tensor using a flattened kernel and bias, and stores the result in a flattened output tensor.
 *
 * @param input Flattened 1D vector representing the input tensor (size: input_height * input_width * input_channels).
 * @param kernel Flattened 1D vector representing the kernel tensor (size: output_channels * kernel_height * kernel_width * input_channels).
 * @param bias Flattened 1D vector representing the bias tensor (size: output_channels).
 * @param output Reference to a flattened 1D vector to store the convolution output.
 * @param input_height Height of the input tensor.
 * @param input_width Width of the input tensor.
 * @param input_channels Number of channels in the input tensor.
 * @param kernel_height Height of the kernel tensor.
 * @param kernel_width Width of the kernel tensor.
 * @param output_channels Number of channels in the output tensor.
 * @param stride Stride value for the convolution.
 * @param padding Padding type ("same" or "valid").
 */

std::vector<float> load_binary_data(const std::string& file_path, size_t num_elements);
void conv2d_1d(const vector<float> &input, 
               const vector<float> &kernel, 
               const vector<float> &bias, 
               vector<float> &output,
               int input_height, int input_width, int input_channels,
               int kernel_height, int kernel_width, int output_channels,
               int stride, const string &padding, string layername);


#endif // CONV2D_H
