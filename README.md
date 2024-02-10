# Deep Hallucination Classification (Kaggle Competition)

This project focuses on implementing different machine learning techniques to solve a classification task for images hallucinated by deep generative models.
For datasets and other informations you can checkout the official description of the contest [here](https://www.kaggle.com/competitions/unibuc-dhc-2023/overview).

## Datasets
At the beggining of the competition we recieved a set of 12,000 image files for training and 10,000 image files for validation, along with the according labels. The test set was composed of 5,000 image files, which we had to classify into 96 possible classes.

## Model Development

### KNN
I started with a simple K-Nearest Neighbors (__KNN__) model, for which yielded an accuracy score of __21.1%__. I used the model by importing it from the _scikit-learn_ library, and then I just adjusted parameters such as the number of neighbors and the distance computation method to optimize its performance.

### CNN
Furthermore, to achieve improved performance, I designed, implemented, and trained two different Convolutional Neural Network (__CNN__) models using _TensorFlow_ and _Keras_ libraries. These models were designed to leverage the hierarchical pattern recognition capabilities inherent in CNNs, enabling them to capture complex features in the image data and achieve better classification accuracy.

### Data preprocessing
To start, for data preprocessing, I iterated over each image one by one and __resized__ it to ensure that all images would have the same size as the input parameter in the first convolutional layer (__64x64__). Then, I transformed each image into a vector of pixels, which I __normalized__ by dividing the values within it by 255.0, so that each pixel would be within the range of 0.0 and 1.0. Additionally, within the vectors storing the labels, I added their integer values. This preprocessing of labels is necessary to use the softmax activation function in the final fully connected layer of the model.

### Data augmentation
During the augmentation process, all images are __flipped both vertically and horizontally__. Additionally, I simulated a function where a random uniform value between 0 and 1 is generated. If this value is greater than a given probability, a rectangle of __randomly__ obtained dimensions is __erased__ from within the image. More precisely, based on the area of the image and two random values representing an aspect ratio (L:l) and a constant by which to multiply the area, we determine a portion within the image to replace with other random generated pixels.

### First CNN architecure
This architecture consists of a total of __six convolutional layers__ and __three dense layers__, with the last one being used for the final classification of the images. This model is implemented using _Sequential_, which facilitates connecting the neural network layers but also imposes a constraint that each layer accepts only one input and produces only one output.

1. __Input Layer__: input shape of (64, 64, 3) for RGB images of size 64x64 pixels.
2. __Convolutional Layers__: use __ReLU activation function__ and __filters__ with different sizes.
3. __Batch Normalization__: applied to stabilize the training process.
4. __Dropout Layers__:  two are included in the network to prevent overfitting (certain neurons are randomly excluded during training).
5. __Max Pooling Layer__: added after every two convolutional layers to downsample the feature maps.
6. __Fully Connected Layers__: to those I applied kernel regularization to mitigate overfitting.
7. __Output Layer__: consists of 96 units, representing the number of classes, and uses __softmax activation function__ to convert raw predictions into class probabilities.
8. __Compilation__: use __Adam optimizer__ with weight decay and __categorical cross-entropy__ loss function.
9. __Callbacks__: include a checkpoint to save the best model obtained after each training epoch, an early stopping mechanism to halt training if the validation accuracy does not improve for a given number of epochs epochs, and a learning rate scheduler to adjust the learning rate based on the current epoch.

![Image](imgs/cnn_architecture.svg)

Using this model I achieved an accuracy score of __93.6%__.

### Second CNN architecure

Towards the end of the competition, I attempted to implement a network that uses skip connections. By concatenating the output of the second convolutional layer with the sixth one, before applying the final dropout layer, the model is able to capture features learned the initial stages and use them later on.

To add this __skip layers format__, I modified the architecture of the model and abandoned the use of _Sequential_, and implemented a model from the _Model class_. For this, I initialized a constructor to include the necessary layers, and unified them within the _call_ method, which is a standard method that I have overridden.
To concatenate the two layers, I had to perform __down-sampling__ by reducing the size of the __second convolutional layer__ through a MaxPooling2D layer with a size of __(8x8)__.

This tested model contains the same layers as the one described above, along with the same defined callbacks (checkpoint, early stopping, and learning rate scheduler).
However, I do not consider it a successful model as it __tends to overfit__.

## Model Evaluation
The performance of our trained models was evaluated using the classification accuracy metric on a separate test set.

## Project Structure
1. _documentation.pdf_ - contains the documentation for this univeristy project, written in romanian
2. _knn_classifier.py_ - the code that I used to implement the knn classifier
3. _cnn_classifier.py_ - the code that I used to implement the cnn classifier, without the skip layers
4. _cnn_classifier_skip.py_ - the code that I used to implement the cnn classifier, with the skip layers structure




