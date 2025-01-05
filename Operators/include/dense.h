#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <array>
#include <string>

void dense(const std::vector<float>& input, const std::vector<float>& weights,
           const std::vector<float>& bias, std::vector<float>& output,
           const std::array<int, 2>& input_shape, const std::array<int, 2>& output_shape,
           const std::string& activation, std::string layername);

void softmax(std::vector<float>& tensor);

#endif // DENSE_H