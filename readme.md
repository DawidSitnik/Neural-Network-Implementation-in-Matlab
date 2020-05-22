# Clothes Recognition with ANN

## Project Purpose
The project aims to make clothes classification with the usage of a self-implemented artificial neural network and its modified version.The modification implemented to neural networks was based on Yann LeCun's article named 'Efficient Backprop'.

## Artificial Naural Network Classification
The input layer of the network consists of 784 neurons - each represents one pixel of the cloth image whose size is 28x28. 
Each neuron of the last layer represents the probability of corresponding to its class. For the classification of the image, the class with the highest probability is chosen.

## Neural Network Training 
The training process relies on calculating the gradient of the loss function (which is the modulo of the difference between obtained values for actual weights and the desirable values) on weights. The desirable values are 1 for the output which represents an actual class and -1 for the rest. The stochastic learning was implemented, so after the calculation of the gradient for each sample, the values of the weights are modified oppositely to the direction of the gradient. The equation which describes weights change is: 

<p align="center">
  <img src = "https://imgur.com/8ylBwIl.png"/>
</p> 

W - set of all the weights, n - learning rate

The model was trained during 200 epochs, for 100 hidden layers and learning rate = 0.001. An error after each epoch can be seen at the plot below. The referred error of value of 12.58% on the testing dataset was obtained around 40 epoch.

<p align="center">
  <img src = "https://imgur.com/2KicsP5.png"/>
</p> 

*Total Error, after 200 epochs:*
- training dataset: 5.38%
- testing dataset: 11.47%


What we can gather from the picture is that an error of the testing set slightly continuously diminishes. It means that theoretically the network could be trained even better. What's more, the overfitting doesn't bring a huge difference between training and testing sets (at least up to 40th epoch), so the architecture of the network could be more complex, which would potentially positively affect the result. 


## Modified Version of the ANN
The next part of the exercise is to implement some refinements basing on Yann LeCun's article named 'Efficient Backprop'.

### Floating Learning Rate
Learning rate describes how fast does the neural network learn, so the bigger it is, the network learns faster. The downside of setting the learning rate to bigger values is that the model will not notice the local minimum and bypass it. To fix it to the optimal value, LeCun proposes to make it dependent on the length of the learning process. So at the beginning the learning rate should be relatively big and its value should diminish by the time. 

To implement this idea the learning rate was subordinate to the total error of the network in the linear relation. To get the solution that is more accurate than the referential one, the initial learning rate (assuming that the first error value is 0.9) was set to 0.005 and decreases to 0.0005 during the learning process.

The formula which is used for fixing the learning rate is:
learning_rate = 0.0005 + (0.005 − 0.0005)(error − 0.01)/(0.9 − 0.1)

### Including Inertia
While learning process with the constant gradient value and its direction (pointing to the local minimum) and without floating learning rate, the network does steps of the constant value. The article suggests to include inertia in the learning process so that the gradient with the constant direction will increase the speed of weights modification. 

The formula for the inertia can be described as follows:
<p align="center">
  <img src = "https://imgur.com/oeNUzcT.png"/>
</p> 

### Another Tried Modifications
Another modification which was tried, but resulted in worse output was input normalization and modification of demanded values. 

#### Demanded Values Modification
According to the article, the optimal output value is the value, for which the second derivative of the activation function is the greatest. It enables the best usage of non-linear structure and prevents saturation of the values. 

Saturation of the values is an unwanted phenomenon based on pursuing output values to the asymptotic values. Because of that, the values of the weights go to infinity, slowing the learning process. 

#### Input Normalization
The way of normalization proposed in the article was to modify the data in the way in which their mean equals to 0. The standard deviation of the samples should be similar to avoid the domination of one sample in the dataset. In the case of given data, the standard deviation was similar among the samples, so the only modification which was testes was subducting the mean value of the pixels of the training dataset. 

### Result of Modified Network
The neural network after modifications meets the requirements of the task - it gives a better result than its referential not modified version. Similarly to the previous version, it is not that much overtrained so it gives a chance to increase the number of hidden layers, which would make the result even better. The learning process of the net can be seen in the picture below: 
<p align="center">
  <img src = "https://imgur.com/NcekZAj.png"/>
</p> 

The total error for 200 epochs and 100 hidden layers is:
- *for training dataset* - 4.5%*
- *for testing dataset* - 11.3%*

