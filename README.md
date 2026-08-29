# MNIST Digit Recognition

A web service that recognizes handwritten digits(0~9) using a CNN model trained on MNIST dataset.
The deep learning model is implemented based on "ゼロから作るDeep Learing"(斎藤 康毅, O'Reilly Japan).

** Demo: [https://mnist-digit-recognition-w00x.onrender.com] **

> Because this service is hosted on a free plan, it goes into sleep if there is no access for a while. It may take 30 seconds to a minute to start up application on first access.

## About this service

I implemented convolutional layer, pooling layer, and backpropagation using only NumPy without relying on machine learing frameworks like PyTorch or TensorFlow.

## How to use

1. Draw a digit on the canvas.
2. Click the "Predict" button.
3. The prediction result and the confidence level for each digit will be displayed.

