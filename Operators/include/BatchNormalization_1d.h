#ifndef BATCHNORMALIZATION_1D_H
#define BATCHNORMALIZATION_1D_H

#include <vector>

// Function to perform batch normalization for 1D input
void batch_normalization_1d(const std::vector<float>& input, 
                            std::vector<float>& output,
                            const std::vector<float>& gamma, 
                            const std::vector<float>& beta,
                            const std::vector<float>& moving_mean, 
                            const std::vector<float>& moving_variance,
                            float epsilon, size_t channels, size_t height, size_t width);

void batch_normalization_1d1(const std::vector<float>& input, std::vector<float>& output,
                            const std::vector<float>& gamma, const std::vector<float>& beta,
                            const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                            float epsilon, size_t channels);;

#endif // BATCHNORMALIZATION_1D_H
