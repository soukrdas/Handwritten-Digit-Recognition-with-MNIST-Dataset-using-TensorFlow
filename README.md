# Handwritten Digit Recognition system using the MNIST dataset

## Project Overview
This project implements a neural network model to classify handwritten digits from the MNIST dataset using TensorFlow. The goal is to achieve accurate predictions of handwritten digits (0-9) by training a neural network on image data. This project demonstrates a basic deep learning workflow, including data preprocessing, model training, evaluation, and prediction visualization.

## Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a well-known dataset in the field of computer vision, consisting of 70,000 grayscale images of handwritten digits. Each image is 28x28 pixels, with 60,000 images for training and 10,000 for testing.

## Objectives
The primary objectives of this project are:
- Build a neural network model to recognize handwritten digits.
- Train and evaluate the model to achieve high accuracy on the test dataset.
- Visualize predictions to understand the model's performance.

## Model Architecture
The model is a basic neural network built using Keras with TensorFlow as the backend. It consists of:
1. **Flatten Layer**: Converts 2D image arrays into a 1D vector.
2. **Dense Layer (128 neurons, ReLU activation)**: Helps the model learn complex patterns in the data.
3. **Output Layer (10 neurons, Softmax activation)**: Outputs probabilities for each digit (0-9).

## Methodology

### 1. Data Preprocessing
- **Normalization**: Scales pixel values to a range of 0 to 1 by dividing by 255.
- **Data Split**: The dataset is split into training and testing sets (pre-defined in the MNIST dataset).

### 2. Model Training
- **Loss Function**: `sparse_categorical_crossentropy` is used for multiclass classification.
- **Optimizer**: `Adam`, which adapts the learning rate during training for efficient convergence.
- **Metrics**: `accuracy` is tracked to evaluate model performance.

### 3. Model Evaluation
The trained model is evaluated on the test set to determine accuracy and overall performance.

### 4. Prediction and Visualization
Random test images are displayed alongside model predictions to visually inspect the accuracy of classifications.

## Results
After training for 6 epochs, the model achieves a test accuracy of approximately 98%. This demonstrates that the neural network can effectively recognize handwritten digits.

## Key Insights
- **Digit Classification**: The model is effective for digit classification tasks.
- **Deep Learning in Image Recognition**: Shows how a basic neural network can achieve high accuracy on image data.
- **Visualization**: Predicted vs. actual labels help evaluate model performance and identify areas for improvement.

## Future Work
To further enhance this project, potential improvements include:
- Implementing Convolutional Neural Networks (CNNs) for even better accuracy.
- Testing on a broader range of handwritten datasets for generalization.
- Hyperparameter tuning to optimize model performance.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow and Keras
- Matplotlib
- Numpy

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/soukrdas/Handwritten-Digit-Recognition-with-MNIST-Dataset-using-TensorFlow.git
    ```
2. Install required packages:
    ```bash
    pip install tensorflow matplotlib numpy
    ```

### Running the Project
1. Load and preprocess the dataset.
2. Train the model using the command:
    ```python
    model.fit(x_train, y_train, epochs=6)
    ```
3. Evaluate the model:
    ```python
    model.evaluate(x_test, y_test)
    ```
4. Run predictions and visualize results.

### Example Usage
```python
# To visualize predictions
plt.imshow(x_test[i], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[i])}, True: {y_test[i]}")
plt.show()
