# cmake_minimum_required(VERSION 3.10)
# project(Conv2DExample)

# set(CMAKE_CXX_STANDARD 17)


# file(GLOB_RECURSE SRC_FILES Test_Operators/src/*.cpp)
# add_executable(LayerProcessor ${SRC_FILES})

# cmake_minimum_required(VERSION 3.10)

# # Project name
# project(Conv2D)

# # Set C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Include directories
# include_directories(Operators/include)

# # Source files
# file(GLOB SOURCES
#     src/main.cpp
#     Operators/src/conv2d.cpp
# )

# # Add executable
# add_executable(Conv2D ${SOURCES})

# cmake_minimum_required(VERSION 3.10)
# project(KernelBiasShape)

# set(CMAKE_CXX_STANDARD 17)

# # Include directories
# include_directories(F:/MCW/Assignment 2/vcpkg/installed/x64-windows/include)

# # Add the executable
# add_executable(kernel_and_bias_shape "F:/MCW/c++ application/Project_Root/Test_Operators/src/kernel and bias shape.cpp")

# # Link necessary libraries
# target_link_libraries(kernel_and_bias_shape stdc++fs) # Linking with filesystem library for older GCC versions

cmake_minimum_required(VERSION 3.10)

# Project name
project(Conv2DProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/Operators/include
    ${PROJECT_SOURCE_DIR}/Test_Operators/include
    ${PROJECT_SOURCE_DIR}/utilities/include
    ${PROJECT_SOURCE_DIR}/include
)


# Source files
file(GLOB_RECURSE OPERATORS_SRC
    ${PROJECT_SOURCE_DIR}/Operators/src/*.cpp
)

file(GLOB_RECURSE TEST_OPERATORS_SRC
    ${PROJECT_SOURCE_DIR}/Test_Operators/src/*.cpp
)

file(GLOB_RECURSE UTILITIES_SRC
    ${PROJECT_SOURCE_DIR}/utilities/src/*.cpp
)

file(GLOB_RECURSE MAIN_SRC
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Add executable
add_executable(Conv2DApp
    ${OPERATORS_SRC}
    ${TEST_OPERATORS_SRC}
    ${UTILITIES_SRC}
    ${MAIN_SRC}
)

# Link necessary libraries
# find_package(nlohmann_json REQUIRED)
# target_link_libraries(Conv2DApp nlohmann_json::nlohmann_json)

