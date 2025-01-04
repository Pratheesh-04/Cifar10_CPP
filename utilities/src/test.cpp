#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#include <sstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std;

// Function to read a binary file into a vector
template<typename T>
vector<T> read_binary_file(const string& file_path) {
    ifstream file(file_path, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + file_path);
    }

    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);

    vector<T> buffer(file_size / sizeof(T));  // Adjust size for type
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    return buffer;
}

// Function to write a vector to the output file (for debugging)
template<typename T>
void write_vector(ofstream& output_file, const vector<T>& vec) {
    for (const auto& val : vec) {
        output_file << val << " ";
    }
    output_file << endl;
}

// Function to create a proper file path using filesystem::path
fs::path get_full_path(const fs::path& base_dir, const string& file_path) {
    return base_dir / file_path;  // Use filesystem path concatenation
}

// Function to convert backslashes to forward slashes for display purposes
string normalize_path(const fs::path& file_path) {
    string path_str = file_path.string();
    replace(path_str.begin(), path_str.end(), '\\', '/');  // Replace backslashes with forward slashes
    return path_str;
}

int main1() {
    // Set base path for the data directory
    fs::path base_dir = "F:/MCW/c++ application/Project_Root";

    // Open the output text file
    ofstream output_file("F:/MCW/c++ application/Project_Root/results.txt");
    if (!output_file.is_open()) {
        cerr << "Failed to open output file!" << endl;
        return 1;
    }

    // Open the JSON file
    ifstream input_file("F:/MCW/c++ application/Project_Root/configs/config_file.json");
    if (!input_file.is_open()) {
        cerr << "Failed to open JSON file!" << endl;
        return 1;
    } 

    // Parse the JSON file
    json model;
    input_file >> model;

    // Iterate through layers and process each layer
    for (const auto& layer : model["layers"]) {
        output_file << "Layer: " << layer["layer_name"] << endl;
        output_file << "  Type: " << layer["type"] << endl;

        // Write and process input shape
        auto input_shape = layer["attributes"]["input_shape"];
        output_file << "  Input Shape: ";
        for (const auto& dim : input_shape) {
            output_file << (dim.is_null() ? "null" : to_string(dim.get<int>())) << " ";
        }
        output_file << endl;

        // Write and process output shape
        auto output_shape = layer["attributes"]["output_shape"];
        output_file << "  Output Shape: ";
        for (const auto& dim : output_shape) {
            output_file << (dim.is_null() ? "null" : to_string(dim.get<int>())) << " ";
        }
        output_file << endl;

        // Read input binary file
        fs::path input_file_path = base_dir / layer["input_file_path"].get<string>();  // Correct path concatenation
        cout << "Debug: Input file path: " << normalize_path(input_file_path) << endl;  // Normalize and print path
        output_file << "  Reading input from: " << normalize_path(input_file_path) << endl;
        if (fs::exists(input_file_path)) {
            try {
                auto input_data = read_binary_file<float>(input_file_path.string());
                output_file << "    Input Data (first 10 elements): ";
                write_vector(output_file, vector<float>(input_data.begin(), input_data.begin() + min(input_data.size(), size_t(10))));
            } catch (const exception& e) {
                output_file << "    Error reading input data: " << e.what() << endl;
            }
        } else {
            output_file << "    Error: Input file does not exist." << endl;
        }

        // Read weights binary files
        for (const auto& weights_file_path : layer["weights_file_paths"]) {
            fs::path weight_file_path = base_dir / weights_file_path.get<string>();  // Correct path concatenation
            cout << "Debug: Weight file path: " << normalize_path(weight_file_path) << endl;  // Normalize and print path
            output_file << "  Reading weights from: " << normalize_path(weight_file_path) << endl;
            if (fs::exists(weight_file_path)) {
                try {
                    auto weights_data = read_binary_file<float>(weight_file_path.string());
                    output_file << "    Weights Data (first 10 elements): ";
                    write_vector(output_file, vector<float>(weights_data.begin(), weights_data.begin() + min(weights_data.size(), size_t(10))));
                } catch (const exception& e) {
                    output_file << "    Error reading weights data: " << e.what() << endl;
                }
            } else {
                output_file << "    Error: Weights file does not exist." << endl;
            }
        }

        // Read output binary file
        fs::path output_file_path = base_dir / layer["output_file_path"].get<string>();  // Correct path concatenation
        cout << "Debug: Output file path: " << normalize_path(output_file_path) << endl;  // Normalize and print path
        output_file << "  Reading output from: " << normalize_path(output_file_path) << endl;
        if (fs::exists(output_file_path)) {
            try {
                auto output_data = read_binary_file<float>(output_file_path.string());
                output_file << "    Output Data (first 10 elements): ";
                write_vector(output_file, vector<float>(output_data.begin(), output_data.begin() + min(output_data.size(), size_t(10))));
            } catch (const exception& e) {
                output_file << "    Error reading output data: " << e.what() << endl;
            }
        } else {
            output_file << "    Error: Output file does not exist." << endl;
        }

        output_file << endl;  // Add a new line between layers
    }

    // Close the output file
    output_file.close();

    cout << "Results have been written to results.txt." << endl;
    return 0;
}
