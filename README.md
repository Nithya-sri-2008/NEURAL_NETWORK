# NEURAL_NETWORK
# Implementation of Neural Network from Scratch using NumPy

First, I created the letter A, B, C as an array of 0 and 1.  

## Steps

### 1. Creating Labels
- Created the label `Y` which is a NumPy array.

### 2. Importing Libraries
- Imported the libraries of NumPy and Matplotlib for visualization and calculation.
- After importing the library, we visualize the data (e.g., A, B, and C) to check the image size.
- Convert the data labels into NumPy arrays and reshape it to `(1, 30)` because the input layer size should be in the shape of `(1, 30)`.

### 3. Activation Function
Now we have done everything for input. Next is the activation function — we are using **sigmoid function** for activation and its formula is: 1 / (1 + e^-x)

### 4. Forward Propagation
We are implementing forward propagation by creating a function of forward with parameters `X, W1, W2` where:
- `X` is the input data.
- `W1` and `W2` are weights.

Weights are used to determine how much influence an input has on output.  
First, we are writing code for the hidden layer:

```python
Z1 = X.dot(W1)
A1 = sigmoid(Z1)

This applies sigmoid activation to add non-linearity.
Similarly for the output layer:
Z2 = A1.dot(W2)
A2 = sigmoid(Z2)
return A2

