import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import statistics


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
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"



'''
The custom neural network with 2 layers (1 hidden layer). 

How it is implemented: Before doing the train method, preprocessing is done where we flatten the image data
from 28*28 images to 784*1. This helps because now we can have 784 input values coming to our neural network.
The train() method uses backpropagation and the calculations are taken from the class slides. Minibatches are implemented
for faster epoch times and correctly calculating the matrix transpositions we need in backpropagation.
For activation, sigmoid is used.
For loss, MSE (mean squared error) is used.
At prediction stage, the output was a probability distribution, so we added and used findMaxIndex function to transform the 
probablity distribution to a valid output for one-hot-encoding.
'''

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
    def train(self, xVals, yVals, epochs = 3, minibatches = True, mbs = 100):
        for _ in range(epochs):
            inputSize = xVals.shape[0]
            if not minibatches:
                mbs = 1
            x_gen = self.__batchGenerator(xVals, mbs)
            y_gen = self.__batchGenerator(yVals, mbs)
            iterations = int(inputSize / mbs)
            for i in range(iterations):
                x_currBatch = next(x_gen)
                y_currBatch = next(y_gen)
                l1_out, l2_out = self.__forward(x_currBatch)
                l2_errors = self.loss_derivative(y_currBatch, l2_out)
                l2_deltas = l2_errors * self.__sigmoidDerivative(l2_out)
                l1_errors = np.dot(l2_deltas, np.transpose(self.W2))
                l1_deltas = l1_errors * self.__sigmoidDerivative(l1_out)
                l1_adjust = np.dot(np.transpose(x_currBatch), l1_deltas) * self.lr
                l2_adjust = np.dot(np.transpose(l1_out), l2_deltas) * self.lr 
                self.W1 += l1_adjust
                self.W2 += l2_adjust
            print("Epoch {}/{} done.".format(_ + 1, epochs))


    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        prediction_batch = []
        for i in range(layer2.shape[0]):
            index = findMaxIndex(layer2[i])
            prediction = [0] * 10
            prediction[index] = 1
            prediction_batch.append(prediction)
        
        return np.array(prediction_batch)

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

'''
Convolutional neural network implemented by Keras. 

How it is implemented: The CNN built starts with two convolutional layers, then the signal goes through a max pooling layer of 2x2
then it is flattened for processing in fully connected neural networks the end. At first I used sigmoid as an activation but turns
out I only got 9%. I changed the activation from sigmoid to ReLu and it reached 98%. The reason I think ReLu works better is because
it doesn't allow negative values. At the last fully connected layer, we used softmax to get a probability distribution. These
distributions turned into valid outputs using the findMaxIndex helper function.
Cross entropy is used for loss function because we discussed it was a good choice for image processing in class.
Only preprocessing done is turning y values to one-hot encoding. Flattening is only needed when we have a fully connected layer.
'''
def buildCNNModel(xTrain, yTrain):
    model = keras.Sequential()
    inShape = (28,28,1) #Images that is 28x28 pixels.
    lossType = keras.losses.categorical_crossentropy
    opt = tf.train.AdamOptimizer()
    model.add(keras.layers.Conv2D(32, kernel_size = (3,3), activation="relu", input_shape = inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = (3,3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer = opt, loss = lossType)
    #Train model
    model.fit(xTrain.reshape([-1,28,28,1]), yTrain.reshape([-1, 10]), epochs=3, batch_size=100)
    return model

'''
Helper Functions
'''
#Returns the index where the max occurs. Useful in shaping the prediction one-hot encoding.
def findMaxIndex(array):
    max_index = 0
    for i in range(1, len(array)):
        if array[i] > array[max_index]:
            max_index = i
    return max_index

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
    return ((xTrain*MULTIPLIER, yTrain), (xTest*MULTIPLIER, yTest))

#Do flattening here.
def preprocessData(raw):
    reduced_value = reduceRange(raw)
    ((xTrain, yTrain), (xTest, yTest)) = reduced_value        
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    #Flattening for the custom net
    if ALGORITHM == "custom_net":
        xTrain = xTrain.reshape([60000, 784])
        xTest = xTest.reshape([10000, 784])
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
        model = NeuralNetwork_2Layer(inputSize=28*28, neuronsPerLayer=256, outputSize=10)
        model.train(xTrain, yTrain)                #TODO: Write code to build and train your custon neural net.
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        #Build model
        return buildCNNModel(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        #Adding a new dimension for keras.
        preds_returned = model.predict(data.reshape(10000,28,28,1))
        preds_actual = []
        for pred in preds_returned:
            max_index = findMaxIndex(pred)
            result = [0] * 10
            result[max_index] = 1
            preds_actual.append(result)

        return np.array(preds_actual)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    f1_score = getF1Score(preds, yTest)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("F1 score: {}".format(f1_score))

#Calculates f1 score given predictions and expected results.
def getF1Score(preds, yTest):
    c_matrix = np.zeros(shape=(NUM_CLASSES, NUM_CLASSES))
    # p => predicted 
    # t => truth
    for i in range(preds.shape[0]):
        p = findMaxIndex(preds[i]) #Finding which index 1 occurs.
        t = findMaxIndex(yTest[i])
        c_matrix[p, t] += 1

    #Will have (precision, recall) pairs for each class
    calculations = []
    for c in range(NUM_CLASSES):
        true_positives = c_matrix[c][c]
        positives = sum(c_matrix[c]) # Positives = true pos + false pos
        precision = true_positives / positives
        false_negatives = sum([row[c] for row in c_matrix]) - true_positives
        recall = true_positives / (true_positives + false_negatives)
        calculations.append((precision, recall))
    
    #Get average of precision and recall.
    avg_precision = statistics.mean([c[0] for c in calculations])
    avg_recall = statistics.mean([c[1] for c in calculations])

    #Calculate F1 Score
    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
