# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:04:30 2020

@author: JAYAGN
"""

#importing required libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#importing data.
X = pd.read_csv("train_data.csv")
y = pd.read_csv("train_labels.csv")

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=20)

#Sigmoid function squishes the input to (0, 1)
#Softmax also does the same. But, Softmax ensures the sum of the outputs equal to 1.
#In case of our output, we would like to measure what is the probability of the input to each class. 
#for example, if an image has a probability of 0.9 to be digit 2, it is good to have 0.1 probability distributed among the other classes which is done by softmax.
def sigmoid(s):
    return 1/(1 + np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def sigmoid_derv(s):
    return s * (1 - s)

#As we are dealing with Multi-class classification problem, the output will be a probability distribution.
#we have to compare it with our values, which is also a probability distribution and find the error.
#For such scenario, we have to go with cross-entropy as the cost function. 
#Because, cross-entropy function is able to compute error between two probability distributions.
def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

#The Neural Network adds the first hidden layer with 264 neurons(arbitrary), where also the input_dim specifies the size of the input layer. 
#There is only input layer and one hidden layer in the model. Finally, the output layer with 4 neurons.
#The algorithm trains the model with two main processes - Feed-forward and Back-propagation. 
#Feed-forward predicts the output for the given input with some weights
#Back-propagation trains the model by adjusting the weights 
class Neural_Network:
    def __init__(self, x, y):
        self.x = x
        neurons = 264
        self.lr = 1.25
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

#In Feed-forward process the dot product of (input and weights) and adding the bias calculates 'z'. 
#It is passed into next layer, which contains activation functions as mentioned above. 
#This activators produces the output 'a'.
#The output of the current layer will be the input to the next layer and so on.
#In our case, first output of hidden layer will be input to next layer.
#The first layer contains sigmoid activator , the output of which is the input of output layer.
#The output layer has softmax function activator, which produces final output 'a3'.

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z3 = np.dot(self.a1, self.w3) + self.b3
        self.a3 = softmax(z3)

#Back propagation is the calculation of derivatives using chain rule.
#The back-propogation calculates the error from the output of feed-forward. In other words, find the derivative of cost function with respect to weight.
#This error is back-propagated to all the weight matrices by computing gradients in hidden layer , after which weights are updated.
#To know how well our network is performing comparing with the true value, we compute the gradient of the cost function with respect to updates weights and biases 
#We calculate the loss to know, how is model performing in each epoch.

    def backpropogation(self):
        loss = error(self.a3, self.y)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y) 
        z1_delta = np.dot(a3_delta, self.w3.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) 

        self.w3 -= self.lr * np.dot(self.a1.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

#The input has is passed to the feed forward network and output is predicted.
    def prediction(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()
    
model = Neural_Network(np.array(X_train), np.array(y_train))

epochs = 500
for x in range(epochs):
    model.feedforward()
    model.backpropogation()
		
def get_acc(x, y):
    acc = 0
    for i,j in zip(x, y):
        s = model.prediction(i)
        if s == np.argmax(j):
            acc +=1
    return acc/len(x)*100

print("Training accuracy : ", get_acc(np.array(X_train), np.array(y_train)))
print("Test accuracy : ", get_acc(np.array(X_val), np.array(y_val)))