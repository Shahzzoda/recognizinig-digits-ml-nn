"""
Implements a Multilayer Perceptron with Sochastic Gradient Descent
Learning algorithm as seen in Neural Networks and Deep Learning by
Micheal Nielson.
"""

import numpy as np
import random


class Network(object):
    def __init__(self, dimmensions):
        """
            Initializes the network with an array of numbers. Each number in the
            list represents how many neurons should be configured in their respective
            layer. The biases and weights will be set randomly and optimized with the
            learning algorithm over time. Recall that the input layer neurons will not
            have biases.
            Ex:
                Network([2, 12, 4])
                would return a Network object that has 2 nuerons for the input layer,
                one hidden layer with 12 neurons, and 4 neurons in the output layer.
        """
        self.num_layers = len(dimmensions)
        self.dimmensions = dimmensions
        """
            bias -- creates y arrays of size 1 for each layer in dimmensions
            besides 1st one. Each bias is a list of 'array' objects. each
            'array' is a multidimmensional list of size y with arrays of size
            1.
                Ex: to access the the xth value in the yth layer, you would:
                    Network([...]).biases[y-1][x-1][0]
                    For Network([1, 2, 3]).biases, you get:
                    [
                        array(
                            [
                                [a1]
                            ]
                        ),
                        array(
                            [
                                [b1],
                                [b2]
                            ]
                        ),
                        array(
                            [
                                [c1],
                                [c2],
                                [c3]
                            ]
                        )
                    ]
        """
        self.biases = [np.random.randn(y, 1) for y in dimmensions[1:]]
        """
            weights
            -- weights exist for every input a neuron has. It can be visualized
            as the lines that connect the layers of the neurons, each of those lines
            must have weights. Weights will be a list of length len(dimmension) -1 with
            each element as multidimmensional list with y arrays of x items. each element
            in the weights list (is a list that) indicates the weights between each layer,
            and each list indicates the weight between each neuron in that layer.
            Ex: Network([1, 2, 3]) can be visually represented like this:
                    o
                o
            o       o
                o
                    o
            2   6     <- weights between each layer

            so Network([1, 2, 3]).weights will look like this:
            [
                array(
                    [
                        [x1],
                        [x2]
                    ]
                    ),
                array(
                    [
                        [y1, y2],
                        [y3, y4],
                        [y5, y6]
                        ]
                    )
            ]
            where each x & y value is actually an integer from the normal distribution
            with mean of 0 and standard deviation of 1. Each array item can be conceptualized
            as a matrix. If we want to see the matrix of weights between the second and third
            layer, we'd do this: Network([1, 2, 3]).biases[1]. To see the weight between the
            ith and jth neuron, we'd have to access the item in the i-1 th row and j-1 th column,
            like so: Network([1, 2, 3]).biases[1][i-1][j-1]
        """
        self.weights = [np.random.randn(y, x)
                        for y, x in zip(dimmensions[1:], dimmensions[:-1])]

    def SGD(self, training_data, epochs, mini_batch_sizes, eta):
        """
            This is the method respsonsible for the overall analysis. SGD
            stands for stochastic gradient descent, our learning algorithm. We'll be doing this with mini batches.
            Params:
                - training_data: list of tuples representing training
                input, and the desired output
                - epochs: the number of times you'd like to reach 'epoch'
                or completely run through the learning data (in batches).
                - mini_batch_sizes: size of the mini batches we'd train on
                - eta: the learning rate
            Returns:
                nothing.
        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+mini_batch_sizes]
                       for k in range(0, n, mini_batch_sizes)]
            for batch in batches:
                self.update_mini_batch(batch, eta)
            print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, eta):
        """
            Updates to the network's weights and biases to a single batch.
            Params:
                - batch: a list of tuples (x, y) where x is the input and y
                is the desired output. AKA a shuffled section of the
                training data
                - eta: the learning rate, a real number.
            Returns:
                nothing.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # as we cycle through the pairs of training data from the batch
        for x, y in batch:
            # we calculate each change to the nabla we must make
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # and go through the deltas to update (+/-) every nabla we have
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        """
            we go through every xy-pair (test & desired outcome) and see
            what change we have to make for x to be closer to y, called
            the delta_nabla. every time we give xy-pair, there's a new
            correction (delta_nabla) and we keep cumulating these for the
            whole batch. after the batch is fully reviewed, we update the weights and biases of the network using the following formula:
                w_n' = w_n - (eta/m)(summation of C_x)
            where w_n' is the "smarter" weight compared to w_n (by one
            iteration), eta is the learning rate, m is the mini batch size,
            and summation of C_x is sum of all the corrections we've made
            to the cost equation.
            This is the core "learning" aspect of the neural network.
        """
        self.weights = [w - (eta/len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
            Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``.
            Params:
                - x: the test input
                - y: the desired output
            Returns:
                - a tuple of nabla_b and nabla_w that are both multi-
                dimmensional lists in the shape of self.biases and
                self.weights, respectively.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """
            Return the vector of partial derivatives \partial C_x /\partial a for the output activations.
            Params:
                - output_activations: a real value 
                - y: a real value. specifically, the desired output. 
            Returns:
                - Two values being subtracted inside a tuple
        """
        return (output_activations-y)

# Miscellaneous functions


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
