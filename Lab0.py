import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

# Disable some troublesome logging.
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def __getError(self, predicted_y, expected_y):
        return 0.5 * (math.pow(predicted_y, 2) - math.pow(expected_y, 2))

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = False, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        #CurrentLayer will keep track of the layer we are in.
        for _ in range(epochs):
            if minibatches:
                inputSize = xVals.shape[0]
                ind = 0
                while ind < inputSize:
                    l1_out, l2_out = self.__forward(xVals[ind:ind+mbs])
                    l2_errors = self.loss_derivative(yVals[ind:ind+mbs],l2_out)
                    l2_deltas = l2_errors * self.__sigmoidDerivative(l2_out)
                    l1_errors = np.dot(l2_deltas, np.transpose(self.W2))
                    l1_deltas = l1_errors * self.__sigmoidDerivative(l1_out)
                    l1_adjust = np.dot(self.W1, np.array(l1_deltas))
                    l2_adjust = np.dot(self.W2, np.array(l2_deltas))
                    self.W1 = self.W1 + l1_adjust
                    self.W2 = self.W2 + l2_adjust
                    ind += mbs
            else:
                inputSize = xVals.shape[0]
                for ind in range(inputSize):
                    l1_out, l2_out = self.__forward(xVals[ind].flatten())
                    l2_errors = self.loss_derivative(yVals[ind], l2_out)
                    l2_deltas = l2_errors * self.__sigmoidDerivative(l2_out)
                    l1_errors = np.dot(l2_deltas, np.transpose(self.W2))
                    l1_deltas = l1_errors * self.__sigmoidDerivative(l1_out)
                    l1_adjust = np.dot(np.transpose(xVals[ind].flatten()), l1_deltas) * self.lr
                    l2_adjust = np.dot(np.transpose(l1_out), l2_deltas) * self.lr 
                    self.W1 += l1_adjust
                    self.W2 += l2_adjust

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2
    
    def loss(self, y_expected, y_predicted):
        return np.square((y_expected - y_predicted), 2).mean(axis=0) #Mean for every column.

    def loss_derivative(self, y_expected, y_predicted):
        return y_expected - y_predicted



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

def reduceRange(raw):
    MULTIPLIER = 1.0 / 255.0
    xTrain = raw[0][0]
    yTrain = raw[0][1]
    xTest = raw[1][0]
    yTest = raw[1][1]
    return ((xTrain*MULTIPLIER, yTrain*MULTIPLIER), (xTest*MULTIPLIER, yTest*MULTIPLIER))

def preprocessData(raw):
    reduced_value = reduceRange(raw)
    ((xTrain, yTrain), (xTest, yTest)) = reduced_value        
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        model = NeuralNetwork_2Layer(28*28, 10, 10)
        model.train(xTrain, yTrain)                #TODO: Write code to build and train your custon neural net.
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        model = trainModel(data)
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
