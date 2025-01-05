import tensorflow as tf
import numpy as np
import json

# Load JSON configuration
config_file_path = "F:/MCW/c++ application/Project_Root/configs/config_file.json"
with open(config_file_path, "r") as json_file:
    config = json.load(json_file)

# Access layers array from config
layers = config["layers"]

# Base path for weights
base_path = "F:/MCW/c++ application/Project_Root/"

# Initialize a shared variable for layer outputs
layer_output = None

# def print_first_channel_outputs(model, input_data):
#     layer_outputs = [layer.output for layer in model.layers]  # Get outputs of all layers
#     intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
#     # Get outputs for the input data
#     outputs = intermediate_model.predict(input_data)
#     for i, output in enumerate(outputs):
#         print(f"Layer {i+1} - {model.layers[i].name}")
#         if len(output.shape) > 2:  # For layers with multiple channels
#             print(output[..., 0])  # Print only the first channel
#         else:
#             print(output)  # For 1D or scalar outputs, print as is
#         print("-" * 50)

def save_output_to_file(output, layer_name, output_height=32, output_width=32):
    # Check the shape of the output before flattening
    print(output)
    print(f"Shape of the layer output: {output.shape}")

    # Flatten the output into a 1D vector
    flattened_output = output.flatten()
    flattened_output = flattened_output[:output_height*output_width]
    output_str = " ".join(map(str, flattened_output))

    # Save the output string to a text file
    file_path ="F:/MCW/c++ application/Project_Root/data/python_outputs/" + f"{layer_name}_output.txt"
    with open(file_path, 'w') as f:
        f.write(output_str)

    print(f"Output for {layer_name} saved to {file_path}")


# Process each layer sequentially
for layer in layers:
    if layer["layer_name"] == "conv2d":
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 2, "Invalid number of weight paths in the configuration file for conv2d layer"

        kernel_path = base_path + weights_file_paths[0]
        bias_path = base_path + weights_file_paths[1]

        input_height = layer["attributes"]["input_shape"][1]
        input_width = layer["attributes"]["input_shape"][2]
        input_channels = layer["attributes"]["input_shape"][3]

        # Initialize input (if it's the first layer)
        if layer_output is None:
            layer_output = np.ones((1, input_height, input_width, input_channels), dtype=np.float32)  # Dummy values (1.0)

        kernel_height = layer["attributes"]["kernel_size"][0]
        kernel_width = layer["attributes"]["kernel_size"][1]
        output_channels = layer["attributes"]["output_shape"][3]

        # Load weights and biases
        kernels_flat = np.fromfile(kernel_path, dtype=np.float32)
        biases = np.fromfile(bias_path, dtype=np.float32)

        kernels = kernels_flat.reshape((kernel_height, kernel_width, input_channels, output_channels))

        # Perform Conv2D operation with linear activation
        strides = layer["attributes"]["strides"]
        padding = layer["attributes"]["padding"].upper()

        conv2d_output = tf.nn.conv2d(
            input=layer_output,
            filters=kernels,
            strides=[1, strides[0], strides[1], 1],
            padding=padding
        )

        # Add biases to the output
        conv2d_output = tf.nn.bias_add(conv2d_output, biases)

        # Update the shared layer output
        layer_output = conv2d_output.numpy()

        # Display full output for the first channel (channel 0) after Conv2D
        save_output_to_file(layer_output[0], layer["layer_name"],32,32)

    elif layer["layer_name"] == "batch_normalization":
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 4, "Invalid number of weight paths in the configuration file for batch_normalization layer"

        gamma_path = base_path + weights_file_paths[0]
        beta_path = base_path + weights_file_paths[1]
        moving_mean_path = base_path + weights_file_paths[2]
        moving_variance_path = base_path + weights_file_paths[3]

        output_channels = layer["attributes"]["output_shape"][3]

        # Load weights
        gamma = np.fromfile(gamma_path, dtype=np.float32)
        beta = np.fromfile(beta_path, dtype=np.float32)
        moving_mean = np.fromfile(moving_mean_path, dtype=np.float32)
        moving_variance = np.fromfile(moving_variance_path, dtype=np.float32)
        epsilon = 1e-5  # Small constant to avoid division by zero

        # Perform Batch Normalization using TensorFlow's inbuilt function
        layer_output = tf.nn.batch_normalization(
            x=layer_output,
            mean=moving_mean,
            variance=moving_variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon
        )

        # Convert TensorFlow tensor to numpy array for inspection
        layer_output = layer_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],32,32)

    if layer["layer_name"] == "max_pooling2d":
        strides = layer["attributes"]["strides"]
        padding = layer["attributes"]["padding"].upper()

        # Default pool_size for MaxPooling2D
        pool_size = [2, 2]  # Default pool size

        # Perform MaxPooling operation
        maxpooling_output = tf.nn.max_pool2d(
            input=layer_output,
            ksize=[1, pool_size[0], pool_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            padding=padding
        )

        # Update the shared layer output
        layer_output = maxpooling_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],16,16)
    
    elif layer["layer_name"] == "conv2d_1":
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 2, "Invalid number of weight paths in the configuration file for conv2d layer"

        kernel_path = base_path + weights_file_paths[0]
        bias_path = base_path + weights_file_paths[1]

        input_height = layer["attributes"]["input_shape"][1]
        input_width = layer["attributes"]["input_shape"][2]
        input_channels = layer["attributes"]["input_shape"][3]

        kernel_height = layer["attributes"]["kernel_size"][0]
        kernel_width = layer["attributes"]["kernel_size"][1]
        output_channels = layer["attributes"]["output_shape"][3]

        # Load weights and biases
        kernels_flat = np.fromfile(kernel_path, dtype=np.float32)
        biases = np.fromfile(bias_path, dtype=np.float32)

        kernels = kernels_flat.reshape((kernel_height, kernel_width, input_channels, output_channels))

        # Perform Conv2D operation with linear activation
        strides = layer["attributes"]["strides"]
        padding = layer["attributes"]["padding"].upper()

        conv2d_output = tf.nn.conv2d(
            input=layer_output,
            filters=kernels,
            strides=[1, strides[0], strides[1], 1],
            padding=padding
        )

        # Add biases to the output
        conv2d_output = tf.nn.bias_add(conv2d_output, biases)

        # Update the shared layer output
        layer_output = conv2d_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],16,16)
    
    elif layer["layer_name"] == "batch_normalization_1":
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 4, "Invalid number of weight paths in the configuration file for batch_normalization layer"

        gamma_path = base_path + weights_file_paths[0]
        beta_path = base_path + weights_file_paths[1]
        moving_mean_path = base_path + weights_file_paths[2]
        moving_variance_path = base_path + weights_file_paths[3]

        output_channels = layer["attributes"]["output_shape"][3]

        # Load weights
        gamma = np.fromfile(gamma_path, dtype=np.float32)
        beta = np.fromfile(beta_path, dtype=np.float32)
        moving_mean = np.fromfile(moving_mean_path, dtype=np.float32)
        moving_variance = np.fromfile(moving_variance_path, dtype=np.float32)
        epsilon = 1e-5  # Small constant to avoid division by zero

        # Perform Batch Normalization using TensorFlow's inbuilt function
        layer_output = tf.nn.batch_normalization(
            x=layer_output,
            mean=moving_mean,
            variance=moving_variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon
        )

        # Convert TensorFlow tensor to numpy array for inspection
        layer_output = layer_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],16,16)

    elif layer["layer_name"] == "max_pooling2d_1":
        strides = layer["attributes"]["strides"]
        padding = layer["attributes"]["padding"].upper()

        # Default pool_size for MaxPooling2D
        pool_size = [2, 2]  # Default pool size

        # Perform MaxPooling operation
        maxpooling_output = tf.nn.max_pool2d(
            input=layer_output,
            ksize=[1, pool_size[0], pool_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            padding=padding
        )

        
        # Update the shared layer output
        layer_output = maxpooling_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],8,8)

    elif layer["layer_name"] == "conv2d_2":
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 2, "Invalid number of weight paths in the configuration file for conv2d layer"

        kernel_path = base_path + weights_file_paths[0]
        bias_path = base_path + weights_file_paths[1]

        input_height = layer["attributes"]["input_shape"][1]
        input_width = layer["attributes"]["input_shape"][2]
        input_channels = layer["attributes"]["input_shape"][3]

        kernel_height = layer["attributes"]["kernel_size"][0]
        kernel_width = layer["attributes"]["kernel_size"][1]
        output_channels = layer["attributes"]["output_shape"][3]

        # Load weights and biases
        kernels_flat = np.fromfile(kernel_path, dtype=np.float32)
        biases = np.fromfile(bias_path, dtype=np.float32)

        kernels = kernels_flat.reshape((kernel_height, kernel_width, input_channels, output_channels))

        # Perform Conv2D operation with linear activation
        strides = layer["attributes"]["strides"]
        padding = layer["attributes"]["padding"].upper()

        conv2d_output = tf.nn.conv2d(
            input=layer_output,
            filters=kernels,
            strides=[1, strides[0], strides[1], 1],
            padding=padding
        )

        # Add biases to the output
        conv2d_output = tf.nn.bias_add(conv2d_output, biases)

        # Update the shared layer output
        layer_output = conv2d_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],8,8)
    
    elif layer["layer_name"] == "batch_normalization_2":
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 4, "Invalid number of weight paths in the configuration file for batch_normalization layer"

        gamma_path = base_path + weights_file_paths[0]
        beta_path = base_path + weights_file_paths[1]
        moving_mean_path = base_path + weights_file_paths[2]
        moving_variance_path = base_path + weights_file_paths[3]

        output_channels = layer["attributes"]["output_shape"][3]

        # Load weights
        gamma = np.fromfile(gamma_path, dtype=np.float32)
        beta = np.fromfile(beta_path, dtype=np.float32)
        moving_mean = np.fromfile(moving_mean_path, dtype=np.float32)
        moving_variance = np.fromfile(moving_variance_path, dtype=np.float32)
        epsilon = 1e-5  # Small constant to avoid division by zero

        # Perform Batch Normalization using TensorFlow's inbuilt function
        layer_output = tf.nn.batch_normalization(
            x=layer_output,
            mean=moving_mean,
            variance=moving_variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon
        )

        # Convert TensorFlow tensor to numpy array for inspection
        layer_output = layer_output.numpy()
        save_output_to_file(layer_output[0], layer["layer_name"],8,8)
        layer_output = layer_output.flatten()


    if layer["layer_name"] == "dense":
        # Load the weights and biases for the Dense layer
        weights_file_paths = layer["weights_file_paths"]
        assert len(weights_file_paths) == 2, "Invalid number of weight paths in the configuration file for dense layer"

        weights_path = base_path + weights_file_paths[0]
        bias_path = base_path + weights_file_paths[1]

        # Load weights and biases
        weights = np.fromfile(weights_path, dtype=np.float32)
        print(weights.shape)
        biases = np.fromfile(bias_path, dtype=np.float32)

        input_size = layer["attributes"]["input_shape"][1]
        output_size = layer["attributes"]["output_shape"][1]

        # Reshape weights to match the Dense layer configuration
        weights = weights.reshape((input_size, output_size))

        # Flatten the input if necessary
        if len(layer_output.shape) > 2:
            layer_output = layer_output.reshape((layer_output.shape[0], -1))

        # Perform the forward pass through the Dense layer
        dense_output = tf.matmul(layer_output, weights) + biases

        # Apply activation function (e.g., linear, ReLU, etc.)
        activation = layer["attributes"]["activation"]
        if activation == "relu":
            dense_output = tf.nn.relu(dense_output)
        elif activation == "softmax":
            dense_output = tf.nn.softmax(dense_output)
        else:
            pass

        # Update the shared layer output
        layer_output = dense_output.numpy()

        # Display the Dense layer output
        print("===============================================================================")
        print("Dense Layer Output:")
        print(layer_output)
        print("===============================================================================")