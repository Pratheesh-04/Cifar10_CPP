#ifndef UNIT_TEST_H
#define UNIT_TEST_H

#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

std::vector<float> read_file_to_vector(const std::string& file_path);
void unit_test(const std::string& layer, const std::string& output_file, const std::string& expected_file, float epsilon = 0.001);
#endif