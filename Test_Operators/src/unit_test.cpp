#include "unit_test.h"
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>

std::ofstream result_file("F:/MCW/c++ application/Project_Root/report/unit_test_check.txt", std::ios::app);

// Utility function to read a file into a 1D vector
std::vector<float> read_file_to_vector(const std::string& file_path) {
    std::vector<float> data;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return {};
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            data.push_back(value);
        }
    }

    file.close();
    return data;
}

void unit_test(const std::string& layer, const std::string& output_file, const std::string& expected_file, float epsilon) {
    auto output_data = read_file_to_vector(output_file);
    auto expected_data = read_file_to_vector(expected_file);

    if (output_data.size() != expected_data.size()) {
        std::cerr << "Test Failed: Size mismatch between output (" << output_data.size() 
                  << ") and expected (" << expected_data.size() << ") data." << std::endl;
        return;
    }

    bool test_passed = true;
    for (size_t i = 0; i < output_data.size(); ++i) {
        if (std::abs(output_data[i] - expected_data[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": output = " << output_data[i]
                      << ", expected = " << expected_data[i] << std::endl;
            test_passed = false;
        }
    }

    if (test_passed) {
        result_file <<layer<< " Test Passed: All values match within tolerance (" << epsilon << ")." << std::endl;
    } else {
        std::cerr << " Test Failed: Mismatched values found." << std::endl;
    }
}