# Clothes Recognition with ANN

## Project Purpose
The aim of the project is to make clothes classification with the usage of self implemented artificial neural network and its modified version.
The modification implemented to neural networks were based on Yann LeCun article named 'Efficient Backprop'.

## Artificial Naural Network Classification
The input layer of the network consists of 784 neurons - each represents one pixel of the cloth image which size is 28x28. 
Each neuron of the last layer represents probability of corresponding to it class. For the classification of the image, the class with the highest probability is chosen.

## Neural Network Training 
The training process relies on callculating gradient of loss function (which is modulo of the difference between obtained values for actual weights and the desirable values) on weights. The desirabled values are 1 for the output which represent an actual class and -1 for the rest. The stochastic learning was implemented, so after callculation of the gradient for each sample, the values of the weights are modified oppositely to the direction of the gradient. The equation which describes weights change is: 

<p align="center">
  <img src = "https://imgur.com/8ylBwIl.png"/>
</p> 

W - set of all the weights, n - learning rate

To obtain referencial efficiency of the network, the model was trained during 100 epochs, for 100 hidden layers and for n = 0.001. An error after each epoch can be seen at the plot below. 

<p align="center">
  <img src = "https://imgur.com/8ylBwIl.png"/>
</p> 

What we can gather from the picture is that an error of the testing set slightly continously dimishes. It means that teoretically the networ could be trained even better. Whats more, the overfitting doesn't bring a huge difference between training and testing set, so the architecture of the network could also be more complex, which would potentially affect the result in a positive way. 

*Total Error:*
- training dataset: 
- testing dataset: 

## Modified Version of the ANN
The next part of the excercise is to implement some refinements basing on Yann LeCun article named 'Efficient Backprop'.

### Floating Learning Rate
Learning rate describes how fast does the neural network learns, so the bigger it is, the network learns faster. The downside of setting learning rate to bigger values is that the model will not notice local minimum and bypass it. To fix it to the optimal value, LeCun proposes to make it dependent on the length of the learning process. So at the beggining the learning rate should be relatively big and its value should deminish by the time. 

To implement this idea the learning rate was subordinate to the total error of the network in the linear relation. To get the solution which is more accurate than the referential one, the initial learning rate (assuming that an first error value is 0.9) was set to 0.005 and decreases to 0.0005 during the learning process.

The formula which is used for fixing the learning rate is:
learning_rate = 0.0005 + (0.005 − 0.0005)(error − 0.01)/(0.9 − 0.1)

### Data Normalization
Another modification proposed in the article



