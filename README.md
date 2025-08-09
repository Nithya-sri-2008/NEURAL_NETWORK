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
```

### 5. Weight Initialization
Next, we initialize the weights. I named this function as generate_wt.
Inside this function:

Create an empty list l.

We need initial weights before training starts, so we randomly initialize the values with np.random.randn() and append this to the list.

### 6. Loss Function
After initializing the weights, we now take care of the loss.
I used Mean Squared Error:
loss = np.square(out - Y)
This is used to calculate the difference between predicted output and actual output.

### 7. Backpropagation
Then calculate the mean. Next is back propagation.
I named this function as back_prop and parameters are (X, Y, W1, W2, alpha):

Assign the forward propagation output to compute predictions for hidden layer and output layer.

For every error that occurs in output layer, define a variable d2 to calculate that:
d2 = (A2 - out) * derivative
The variable d1 is used to calculate the error for hidden layer.

For gradient calculation and updating the weights:
W1 = old_weight - (learning_rate * W1_adj)
W2 = old_weight - (learning_rate * W2_adj)

### 8. Training
Training our dataset: function is named as train with parameters (X, Y, W1, W2, alpha, epoch).

Alpha is the learning rate and epoch is the iteration count.

We calculate accuracy and loss for each epoch and append it to a list.

### 9. Prediction
Next is the prediction function — it is named as predict(X, W1, W2):

Passes the input X to get the output.

Loops through the output to find the index of largest value:

K = 0 → print letter A

K = 1 → print letter B

K = 2 → print letter C

Reshape the input into (5, 6).

10. Initializing Weights
Next initialize the weights:
W1 = generate_wt(30, 5)
W2 = generate_wt(5, 3)


### 11. Final Step
The final step is passing the arguments to train dataset, which are:

X label

Y label

Weights

Learning rate

Number of epochs

Next:

Plot the graph between the loss and accuracy to find accuracy.

Plot the graph between loss and epochs to find the loss.

At last, use the predict function to get the result:
predict(X[2], W1, W2)





