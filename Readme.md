# Single Layer Neural Network (Boolean Expressions)
#Autor: Каргбо Даниел бартоломео

## Overview
This project demonstrates the implementation and training of a simple **single-layer neural network (NN)** to solve basic Boolean classification problems like **AND** and **OR** using the **Widrow-Hoff learning rule**.


The project aims to study the behavior of the simplest neural network structure, consisting of a single neuron with a non-linear activation function.

---

## Features
- **Activation Function**: Step function to classify inputs.
- **Learning Algorithm**: Widrow-Hoff learning rule for weight and bias updates.
- **Boolean Problems**:
  - AND
  - OR
  - XOR (explored but unsolvable with a single-layer NN).
- **Error Tracking**: Displays total error at each epoch.

---

## Requirements
- Python 3.7+
- Libraries: `numpy`

To install dependencies, run:
```bash
pip install numpy
```

---

## Usage

### 1. Clone or Download the Repository
Save the project files locally.

### 2. Run the Code
Execute the Python file:
```bash
python single_layer_nn.py
```

### 3. Output
The program trains the neural network for both **AND** and **OR** operations and evaluates its performance.

#### Example Output for AND Operation:
```
Training for AND operation:
Epoch 1/100, Total Error: 3
Epoch 2/100, Total Error: 2
...
Epoch 10/100, Total Error: 0

Testing AND operation:
Input: [0 0], Predicted: 0
Input: [0 1], Predicted: 0
Input: [1 0], Predicted: 0
Input: [1 1], Predicted: 1
```

---

## Code Structure

### `SingleLayerNN` Class
- **`__init__`**: Initializes weights, bias, and learning rate.
- **`predict`**: Makes a prediction using the step activation function.
- **`train`**: Trains the model using the Widrow-Hoff learning rule.

### Main Functionality
1. Define training data and labels for AND, OR, and XOR.
2. Train the network for AND and OR operations.
3. Test and display predictions for each operation.

---

## Limitations
- The single-layer NN cannot solve the XOR problem because XOR is not linearly separable. Solving XOR requires a multi-layer neural network.

---

## Further Exploration
- Extend the implementation to a **multi-layer perceptron (MLP)** using libraries like TensorFlow or PyTorch to solve XOR.
- Experiment with different learning rates and epochs to observe their effect on training.

---

## References
1. **Widrow-Hoff Learning Rule**: [Wikipedia](https://en.wikipedia.org/wiki/Least_mean_squares_filter)
2. **Neural Networks and Boolean Functions**: [Educational Resources](https://www.educative.io/)

---

## License
This project is open-source and available under the MIT License.

