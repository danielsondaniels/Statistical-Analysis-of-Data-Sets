import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0
class SingleLayerNN:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return step_function(summation)

    def train(self, training_data, labels, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)
                self.weights += self.learning_rate * error * np.array(inputs)
                self.bias += self.learning_rate * error
            print(f"Epoch {epoch + 1}/{epochs}, Total Error: {total_error}")
training_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels_and = np.array([0, 0, 0, 1])
labels_or = np.array([0, 1, 1, 1])
labels_xor = np.array([0, 1, 1, 0])

print("Training for AND operation:")
nn_and = SingleLayerNN(input_size=2, learning_rate=0.1)
nn_and.train(training_data, labels_and)

print("\nTesting AND operation:")
for inputs in training_data:
    print(f"Input: {inputs}, Predicted: {nn_and.predict(inputs)}")
print("\nTraining for OR operation:")
nn_or = SingleLayerNN(input_size=2, learning_rate=0.1)
nn_or.train(training_data, labels_or)

print("\nTesting OR operation:")
for inputs in training_data:
    print(f"Input: {inputs}, Predicted: {nn_or.predict(inputs)}")


