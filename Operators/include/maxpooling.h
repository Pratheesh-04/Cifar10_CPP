#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include <vector>
#include <array>
#include <string>

// Function declaration for 1D MaxPooling
void max_pooling2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
                   const std::array<int, 2>& pool_size, const std::array<int, 2>& strides,
                   const std::string& padding, std::string layername);

#endif // MAXPOOLING_H
