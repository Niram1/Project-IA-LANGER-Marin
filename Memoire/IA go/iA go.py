# tag::avg_imports[]
import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_doubl
import gzip
import numpy as np


# end::avg_imports[]


# tag::average_digit[]
def average_digit(data, digit):  # <1>
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)  # <2>

# <1> We compute the average over all samples in our data representing a given digit.
# <2> We use the average eight as parameters for a simple model to detect eights.
# end::average_digit[]

# tag::display_digit[]
from matplotlib import pyplot as plt

img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()
# end::display_digit[]

# tag::eval_eight[]
x_3 = train[2][0]    # <1>
x_18 = train[17][0]  # <2>

W = np.transpose(avg_eight)
np.dot(W, x_3)   # <3>
np.dot(W, x_18)  # <4>

# <1> Training sample at index 2 is a "4".
# <2> Training sample at index 17 is an "8"
# <3> This evaluates to about 20.1.
# <4> This term is much bigger, about 54.2.
# end::eval_eight[]


# tag::predict_simple[]
def predict(x, W, b):  # <1>
    return sigmoid_double(np.dot(W, x) + b)


b = -45  # <2>

print(predict(x_3, W, b))   # <3>
print(predict(x_18, W, b))  # <4> 0.96

# <1> A simple prediction is defined by applying sigmoid to the output of np.dot(W, x) + b.
# <2> Based on the examples computed so far we set the bias term to -45.
# <3> The prediction for the example with a "4" is close to zero.
# <5> The prediction for an "8" is 0.96 here. We seem to be onto something with our heuristic.
# end::predict_simple[]


# tag::evaluate_simple[]
def evaluate(data, digit, threshold, W, b):  # <1>
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:  # <2>
            correct_predictions += 1
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:  # <3>
            correct_predictions += 1
    return correct_predictions / total_samples

# <1> As evaluation metric we choose accuracy, the ratio of correct predictions among all.
# <2> Predicting an instance of an eight as "8" is a correct prediction.
# <3> If the prediction is below our threshold and the sample is not an "8", we also predicted correctly.
# end::evaluate_simple[]


# tag::evaluate_example[]
evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)  # <1>
# tag::imports[]
import numpy as np
# end::imports[]


# tag::sigmoid[]
def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)
# end::sigmoid[]


# tag::sigmoid_prime[]
def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)
# end::sigmoid_prime[]


# tag::layer[]
class Layer(object):  # <1>
    def __init__(self):
        self.params = []

        self.previous = None  # <2>
        self.next = None  # <3>

        self.input_data = None  # <4>
        self.output_data = None

        self.input_delta = None  # <5>
        self.output_delta = None
# <1> Layers are stacked to build a sequential neural network.
# <2> A layer knows its predecessor ('previous')...
# <3> ... and its successor ('next').
# <4> Each layer can persist data flowing into and out of it in the forward pass.
# <5> Analogously, a layer holds input and output data for the backward pass.
# end::layer[]

# tag::connect[]
    def connect(self, layer):  # <1>
        self.previous = layer
        layer.next = self
# <1>  This method connects a layer to its direct neighbours in the sequential network.
# end::connect[]

# tag::forward_backward[]
    def forward(self):  # <1>
        raise NotImplementedError

    def get_forward_input(self):  # <2>
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):  # <3>
        raise NotImplementedError

    def get_backward_input(self):  # <4>
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(self):  # <5>
        pass

    def update_params(self, learning_rate):  # <6>
        pass

    def describe(self):  # <7>
        raise NotImplementedError

# <1> Each layer implementation has to provid a function to feed input data forward.
# <2> input_data is reserved for the first layer, all others get their input from the previous output.
# <3> Layers have to implement backpropagation of error terms, that is a way to feed input errors backward through the network.
# <4> Input delta is reserved for the last layer, all other layers get their error terms from their successor.
# <5> We compute and accumulate deltas per mini-batch, after which we need to reset these deltas.
# <6> Update layer parameters according to current deltas, using the specified learning_rate.
# <7> Layer implementations can print their properties.
# end::forward_backward[]


# tag::activation_layer[]
class ActivationLayer(Layer):  # <1>
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)  # <2>

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)  # <3>

    def describe(self):
        print("|-- " + self.__class__.__name__)
        print("  |-- dimensions: ({},{})"
              .format(self.input_dim, self.output_dim))
# <1> This activation layer uses the sigmoid function to activate neurons.
# <2> The forward pass is simply applying the sigmoid to the input data.
# <3> The backward pass is element-wise multiplication of the error term with the sigmoid derivative evaluated at the input to this layer.
# end::activation_layer[]


# tag::dense_init[]
class DenseLayer(Layer):

    def __init__(self, input_dim, output_dim):  # <1>

        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)  # <2>
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]  # <3>

        self.delta_w = np.zeros(self.weight.shape)  # <4>
        self.delta_b = np.zeros(self.bias.shape)

# <1> Dense layers have input and output dimensions.
# <2> We randomly initialize weight matrix and bias vector.
# <3> The layer parameters consist of weights and bias terms.
# <4> Deltas for weights and biases are set to zero.
# end::dense_init[]

# tag::dense_forward[]
    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias  # <1>

# <1> The forward pass of the dense layer is the affine linear transformation on input data defined by weights and biases.
# end::dense_forward[]

# tag::dense_backward[]
    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()  # <1>

        self.delta_b += delta  # <2>

        self.delta_w += np.dot(delta, data.transpose())  # <3>

        self.output_delta = np.dot(self.weight.transpose(), delta)  # <4>

# <1> For the backward pass we first get input data and delta.
# <2> The current delta is added to the bias delta.
# <3> Then we add this term to the weight delta.
# <4> The backward pass is completed by passing an output delta to the previous layer.
# end::dense_backward[]

# tag::dense_update[]
    def update_params(self, rate):  # <1>
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):  # <2>
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):  # <3>
        print("|--- " + self.__class__.__name__)
        print("  |-- dimensions: ({},{})"
              .format(self.input_dim, self.output_dim))
# <1> Using weight and bias deltas we can update model parameters with gradient descent.
# <2> After updating parameters we should reset all deltas.
# <3> A dense layer can be described by its input and output dimensions.
# end::dense_update[]

evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)   # <2>

eight_test = [x for x in test if np.argmax(x[1]) == 8]
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b)  # <3>

# <1> Accuracy on training data of our simple model is 78% (0.7814)
# <2> Accuracy on test data is slightly lower, at 77% (0.7749)
# <3> Evaluating only on the set of eights in the test set only results in 67% accuracy (0.6663)
# end::evaluate_example[]

# tag::encoding[]



def encode_label(j):  # <1>
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# <1> We one-hot encode indices to vectors of length 10.
# end::encoding[]


# tag::shape_load[]
def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]  # <1>

    labels = [encode_label(y) for y in data[1]]  # <2>

    return list(zip(features, labels))  # <3>


def load_data_impl():
    # file retrieved by:
    #   wget https://s3.amazonaws.com/img-datasets/mnist.npz -O code/dlgo/nn/mnist.npz
    # code based on:
    #   site-packages/keras/datasets/mnist.py
    path = 'mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_data():
    train_data, test_data = load_data_impl()
    return shape_data(train_data), shape_data(test_data)


# <1> We flatten the input images to feature vectors of length 784.
# <2> All labels are one-hot encoded.
# <3> Then we create pairs of features and labels.
# <4> Unzipping and loading the MNIST data yields three data sets.
# <5> We discard validation data here and reshape the other two data sets.
# end::shape_load[]

# tag::mse[]
import random
import numpy as np


class MSE:  # <1>

    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff * diff)[0]  # <2>

    @staticmethod
    def loss_derivative(predictions, labels):
        return predictions - labels

# <1> We use mean squared error as our loss function.
# <2> By defining MSE as 0.5 times the square difference between predictions and labels...
# <3> ... the loss derivative is simply: predictions - labels.
# end::mse[]


# tag::sequential_init[]
class SequentialNetwork:  # <1>
    def __init__(self, loss=None):
        print("Initialize Network...")
        self.layers = []
        if loss is None:
            self.loss = MSE()  # <2>

# <1> In a sequential neural network we stack layers sequentially.
# <2> If no loss function is provided, MSE is used.
# end::sequential_init[]

# tag::add_layers[]
    def add(self, layer):  # <1>
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

# <1> Whenever we add a layer, we connect it to its predecessor and let it describe itself.
# end::add_layers[]

# tag::train[]
    def train(self, training_data, epochs, mini_batch_size,
              learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):  # <1>
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for
                k in range(0, n, mini_batch_size)  # <2>
            ]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)  # <3>
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(test_data), n_test))  # <4>
            else:
                print("Epoch {0} complete".format(epoch))

# <1> To train our network, we pass over data for as many times as there are epochs.
# <2> We shuffle training data and create mini-batches.
# <3> For each mini-batch we train our network.
# <4> In case we provided test data, we evaluate our network on it after each epoch.
# end::train[]

# tag::train_batch[]
    def train_batch(self, mini_batch, learning_rate):
        self.forward_backward(mini_batch)  # <1>

        self.update(mini_batch, learning_rate)  # <2>

# <1> To train the network on a mini-batch, we compute feed-forward and backward pass...
# <2> ... and then update model parameters accordingly.
# end::train_batch[]

# tag::update_ff_bp[]
    def update(self, mini_batch, learning_rate):
        learning_rate = learning_rate / len(mini_batch)  # <1>
        for layer in self.layers:
            layer.update_params(learning_rate)  # <2>
        for layer in self.layers:
            layer.clear_deltas()  # <3>

    def forward_backward(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()  # <4>
            self.layers[-1].input_delta = \
                self.loss.loss_derivative(self.layers[-1].output_data, y)  # <5>
            for layer in reversed(self.layers):
                layer.backward()  # <6>

# <1> A common technique is to normalize the learning rate by the mini-batch size.
# <2> We then update parameters for all layers.
# <3> Afterwards we clear all deltas in each layer.
# <4> For each sample in the mini batch, feed the features forward layer by layer.
# <5> Next, we compute the loss derivative for the output data.
# <6> Finally, we do layer-by-layer backpropagation of error terms.
# end::update_ff_bp[]

# tag::eval[]
    def single_forward(self, x):  # <1>
        self.layers[0].input_data = x
        for layer in self.layers:
                layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):  # <2>
        test_results = [(
            np.argmax(self.single_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
# <1> Pass a single sample forward and return the result.
# <2> Compute accuracy on test data.
# end::eval[]