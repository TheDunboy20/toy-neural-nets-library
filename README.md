# Toy Neural Nets Library

This project is a Java-based implementation of neural network fundamentals, developed primarily for educational purposes. It provides a simple, layer-based architecture to demonstrate how neural networks process information, perform forward passes, and update through backpropagation.

## Purpose

The primary goal of this library is to serve as an educational resource for understanding the internal mechanics of neural networks. It is designed to be readable and easy to follow, making it a "toy" library rather than a production-ready framework.

## Features

- **Layer-Based Architecture**: Build networks by stacking layers with customizable output sizes.
- **Activation Functions**:
    - ReLU
    - Sigmoid
    - Linear
- **Loss Functions**:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Binary Cross-Entropy (BCE)
    - Categorical Cross-Entropy (CCE)
- **Training Mechanisms**:
    - Forward propagation.
    - Backpropagation for hidden and output layers.
    - Gradient descent for updating weights and biases.

## Project Structure

- `Net`: The main container for the neural network, managing layers and training parameters.
- `Layer`: A collection of perceptrons representing a single layer in the network.
- `Perceptron`: The fundamental unit that performs weighted sums and applies activation functions.
- `activations`: Package containing various activation function implementations.
- `losses`: Package containing various loss function implementations.

## Usage Example

The following example demonstrates how to initialize a simple neural network, add layers, and perform a forward pass with loss calculation.

```java
import msmarik.Layer;
import msmarik.Net;
import msmarik.activations.Activations;
import msmarik.losses.Losses;

public class Example {
    public static void main(String[] args) {
        double[] input = {1.0, 1.0, 1.0, 1.0};
        double[] label = {0.0};

        // Initialize network with MSE loss and learning rate
        Net net = new Net(Losses.MSE(), 0.001);

        // Add layers with specified output sizes and activation functions
        net.addLayer(new Layer(10, Activations.relu()));
        net.addLayer(new Layer(20, Activations.relu()));
        net.addLayer(new Layer(1, Activations.linear()));

        // Perform forward pass
        double[] result = net.forward(input);

        // Calculate loss
        double loss = net.calculateLoss(result, label);

        System.out.println("Neural network result: " + java.util.Arrays.toString(result));
        System.out.println("Calculated loss: " + loss);
    }
}
```

## Setup

This project uses Gradle. You can build it using the provided wrapper:

```bash
./gradlew build
```
