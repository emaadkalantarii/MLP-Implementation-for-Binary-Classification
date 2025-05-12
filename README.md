# Knowledge Discovery and Data Mining Project: Multilayer Perceptron (MLP) Implementation

## Project Overview

This project implements a Multilayer Perceptron (MLP) from scratch using Python.  It was developed as part of a Knowledge Discovery and Data Mining course. The MLP is designed for binary classification tasks. The implementation includes:

* Forward propagation
* Backward propagation
* Weight updates
* Sigmoid and Softmax activation functions
* Batch training
* Loss calculation
* Evaluation metrics (accuracy and confusion matrix)
* Hyperparameter tuning

## Code Description

The core of the project is within the `Assignment__002_B2.ipynb` Jupyter Notebook. Here's a breakdown of the key components:

**1. Libraries**

The following libraries are used:

* `numpy`: For numerical computations.
* `pandas`: For data manipulation (reading Excel files, creating dataframes).
* `matplotlib.pyplot`: For plotting (e.g., the heatmap).
* `seaborn`: For creating the heatmap visualization.

**2. Data Loading and Preprocessing**

* Reads training and validation data from Excel files (`THA2train.xlsx`, `THA2validate.xlsx`).
* Separates features (X) and labels (y) for both training and validation sets.
* One-hot encodes the labels using `pd.get_dummies`.
* Normalizes the features using mean and standard deviation.

**3. Activation Functions**

* `sigmoid(z)`: Computes the sigmoid function, handling potential overflow issues.
* `sigmoid_derivative(a)`: Computes the derivative of the sigmoid function.
* `softmax(z)`: Computes the softmax function for the output layer.

**4. MLP Class**

The `MLP` class encapsulates the neural network:

* `__init__(self, input_size, hidden_size, output_size)`: Initializes the weights and biases for the input, hidden, and output layers.  Weights are initialized to zero.
* `forward(self, X)`: Performs forward propagation.  Calculates the weighted sums and applies the sigmoid and softmax activation functions.
* `backward(self, X, y)`: Performs backward propagation.  Calculates the gradients of the weights and biases with respect to the loss.
* `update_weights(self, dW1, db1, dW2, db2, learning_rate)`: Updates the weights and biases using the calculated gradients and the learning rate.
* `compute_loss(self, y_true, y_pred)`: Computes the cross-entropy loss.

**5. Helper Functions**

* `train_model(model, X_train_norm, y_train, X_validate_norm, y_validate, epochs, learning_rate)`:
    * Implements the training loop.
    * Handles batching, shuffling, forward and backward passes, and weight updates.
    * Calculates and stores training and validation losses.
    * Prints the training and validation loss every 10 epochs.
* `calculate_accuracy(true_labels, predicted_labels)`: Computes the accuracy of the model's predictions.
* `create_confusion_matrix(true_labels, predicted_labels, num_classes)`:  Calculates the confusion matrix.

**6. Model Training and Evaluation**
* Normalizes the training and validation data.
* Instantiates the `MLP` model.
* Defines hyperparameters: `input_size`, `hidden_size`, `output_size`, `epochs`, `batch_size`, `learning_rates`, and `weight_init_params`.  Note that the code as provided only uses a single learning rate; the lists of learning rates and weight initialization parameters are not used in the main training loop.
* The validation set's class labels are extracted.
* The model is trained using the `train_model` function.
* The model's predictions on the validation set are generated.
* The accuracy and confusion matrix are calculated and printed for the validation set.
* A heatmap of validation accuracies for different learning rates and weight initialization parameters is generated.

## Datasets
The project uses two datasets:
* `THA2train.xlsx`: The training dataset.
* `THA2validate.xlsx`: The validation dataset.

These datasets are assumed to be located in a `DataSets` directory in the same directory as the notebook.  Each dataset should contain the features and the target variable for the binary classification task.

## How to Run the Code

1.  **Clone the repository:** Clone this repository to your local machine.
2.  **Install dependencies:** Install the required Python libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn
    ```
3.  **Place the datasets:** Ensure that the `THA2train.xlsx` and `THA2validate.xlsx` files are located in the `DataSets` directory.  Create the directory if it does not exist.
4.  **Run the notebook:** Open and run the `MLPBinaryClassification.ipynb` Jupyter Notebook.
