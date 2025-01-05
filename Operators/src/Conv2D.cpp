#include "conv2d.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <nlohmann/json.hpp> // JSON library
#include <chrono>           // For timing
#include <iomanip>          // For formatted output

using json = nlohmann::json;
using namespace std;

vector<float> load_binary_data(const string &file_path, size_t size) {
    ifstream file(file_path, ios::binary);
    assert(file.is_open() && "Unable to open file");

    vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    file.close();

    return data;
}

void conv2d_1d(const vector<float> &input, 
               const vector<float> &kernel, 
               const vector<float> &bias, 
               vector<float> &output,
               int input_height, int input_width, int input_channels,
               int kernel_height, int kernel_width, int output_channels,
               int stride, const string &padding, string layername) {

    // Calculate padding
    if (padding != "same" && padding != "valid") {
        cerr << "Error: Unsupported padding type: " << padding << endl;
        exit(EXIT_FAILURE);
    }

    int pad_height = 0, pad_width = 0;
    if (padding == "same") {
        pad_height = (kernel_height - 1) / 2;
        pad_width = (kernel_width - 1) / 2;
    }

    // Calculate output dimensions
    int output_height = (input_height + 2 * pad_height - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * pad_width - kernel_width) / stride + 1;
    int output_size = output_height * output_width * output_channels;

    // Resize the output vector
    output.resize(output_size, 0.0f);

    // Start measuring execution time
    auto start_time = chrono::high_resolution_clock::now();

    // Perform convolution
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            for (int c = 0; c < output_channels; ++c) {
                float value = 0.0f;

                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        for (int ic = 0; ic < input_channels; ++ic) {
                            int ih = h * stride + kh - pad_height;
                            int iw = w * stride + kw - pad_width;

                            // Check bounds for valid input
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_idx = ((ih * input_width + iw) * input_channels) + ic;
                                int kernel_idx = ((kh * kernel_width + kw) * input_channels + ic) * output_channels + c;

                                assert(input_idx >= 0 && input_idx < input.size());
                                assert(kernel_idx >= 0 && kernel_idx < kernel.size());

                                value += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }

                // Add bias
                value += bias[c];

                // Store the result in the 1D output vector
                int output_idx = (h * output_width + w) * output_channels + c;
                assert(output_idx >= 0 && output_idx < output.size());
                output[output_idx] = value;
            }
        }
    }

    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> execution_time = end_time - start_time;

    cout << "=====================================================\n";
    // Print output shape
    cout << "Conv2D Output Shape: [" 
         << output_height << ", " 
         << output_width << ", " 
         << output_channels << "]" << endl;

    // Print execution time
    cout << "Execution Time: " << execution_time.count() << " seconds" << endl;

    ofstream outfile("F:/MCW/c++ application/Project_Root/data/cpp_outputs/"+layername+".txt");
    if (outfile.is_open()) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                int output_idx = (h * output_width + w) * output_channels;
                outfile << fixed << setprecision(6) << output[output_idx] << " ";
            }
            outfile << "\n";
        }
        outfile.close();
        cout << "First channel of Conv2D output saved to conv2d_first_channel_output.txt" << endl;
    } else {
        cerr << "Error opening file for writing!" << endl;
    }
    cout << "=====================================================\n";
}
