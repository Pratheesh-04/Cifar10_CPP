# Cifar10_CPP

## Deep Learning Operators and Layers in C++

### Project Overview
This project involves creating a C++ application that handles deep learning model layers, such as Convolutional, ReLU, Softmax, etc. The goal is to implement the core functionality of these layers, including reading inputs, performing computations, and generating outputs. It requires the creation of a C++ application that reads configurations, processes layers, and tests the functionalities.

### Project Folder Contents
#### operators
Implement the C++ version of each required operator (such as Convolution, Batch Normalization, etc.). These are the functions that execute the operations for each layer, such as element-wise addition for a fully connected layer, or the forward pass of a convolution layer. The source folder contains the source files and the include folder contains the header files.

#### Test_Operators
Contains the source code and header files for unit and model testing

#### configs
Contains the json configuration file containing the details of each of the layers of the model.

#### data
Contains the input, output, weights and reference files for each of the layers. Also contains the layer outputs of C++ and python for unit testing.

#### report
Contains the log files for unit testing and model testing.

#### src
Contains the main source file and a .py file for testing purposes.

#### .gitignore
Ensures unnecessary files (e.g., .exe, .lib, .vscode) are not tracked in the Git repository.

#### CMakeLists.txt
A configuration file used by CMake to define how the project should be built, including source files, dependencies, compiler options, and build targets, ensuring platform-independent and scalable builds.
