"""Simple neural network, trains to classify an input array.
Depth and synapse size are parameterized.
"""
import numpy as np

#--------------FUNCTIONS--------------#
def nonlin(x, deriv=False):
    """Sigmoid, takes a number and turns it into a decimal. Set deriv=True for
    the derivative of the sigmoid given the ouput of the sigmoid.
    """
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def cost(error_matrix):
    """Stochastic update loss function. Implicit in the code for backpropagation,
    called explicitly to compare magnitude of errors.
    """
    error = np.array(error_matrix)
    cost = 0.5*(error.norm()**2)
    return cost


def make_syns(x, y, num_layers, synapse_size):
    """Creates random synapses, creating as many as requested to the requested
    size, where synapse_size is the length of one side of a square synapse.
    """
    np.random.seed(1)
    syns = []
    if num_layers==2:
        syns.append(2*np.random.random((len(x[1]), 1))-1)
    else:
        syns.append(2*np.random.random((len(x[1]), synapse_size))-1)
    for i in range(1, num_layers-1):
        if i == num_layers-2:
            syns.append(2*np.random.random((synapse_size, len(y[0])))-1)
        else:
            syns.append(2*np.random.random((synapse_size, synapse_size))-1)
    return syns

def run_once(x, syns, num_layers):
    """Given an input and a list of synapses, calculates an output. Can be used
    for prediction or for training.
    """
    layers = []
    layers.append(x)
    for j in range(1, num_layers):
        layers.append(nonlin(np.dot(layers[j-1], syns[j-1])))
    return layers

def update_syns(x, y, layers, alpha, syns, num_layers):
    """Finds final error and delta, then does backpropagation to adjust previous
    layers. Uses calculated updates to change synapses.
    """
    #Find error
    errors = [None]*(num_layers-1)
    deltas = [None]*(num_layers-1)
    errors[-1] = layers[-1] - y
    deltas[-1] = errors[-1]*nonlin(layers[-1], True)


    #backpropogate to find other errors
    for j in range(2, num_layers):
        errors[-j] = deltas[-j+1].dot(syns[-j+1].T)
        deltas[-j] = errors[-j]*nonlin(layers[-j], True)

    if num_layers==2:
        syns[0] -= alpha*(layers[0].T.dot(deltas[0]))
    else:
        for j in range(num_layers-1):

            syns[j] -= alpha*(layers[j].T.dot(deltas[j]))
    return syns, errors


def train_one_alpha(x, y, num_layers, synapse_size, alpha, num_iter):
    """Trains the neural network once, for one alpha value.
    """
    syns = make_syns(x, y, num_layers, synapse_size)
    for i in range(num_iter):
        layers = run_once(x, syns, num_layers)
        syns, errors = update_syns(x, y, layers, alpha, syns, num_layers)
    print("Output After Training:\n", layers[-1])
    output_error = errors[-1]
    print("Final Error:\n", output_error)
    return layers, syns, output_error

def run_network(x, y, num_layers, synapse_size, alpha,
                num_iter=300000, test_input=None, test_output=None):
    least_training_cost=100000
    training_tuple = (least_training_cost, None)
    least_test_cost=100000
    test_tuple = (least_test_cost, None)

    for alpha in alphas:
        print("Testing alpha value of %s" % (alpha))
        layers, syns, output_error = train_one_alpha(x, y, num_layers, synapse_size, alpha, num_iter)
        if test_input.any():
            test_layers = run_once(test_input, syns, num_layers)
            output = test_layers[-1]
            print("Test:\n", output)
            if test_output.any():
                test_error = test_output-output
                print("Test Error:\n", test_error)

        training_cost = cost(output_error)
        test_cost = cost(test_error)
        if training_cost < least_training_cost:
            training_tuple = (training_cost, alpha)
        if test_cost < least_test_cost:
            test_tuple = (test_cost, alpha)
        








#--------------TRAINING INPUT--------------#
x = np.array([[1, 2, 3, 4, 5],
            [0, 3, 4, 5, 6],
            [2, 4, 6, 7, 8],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4],
            [7, 6, 3, 2, 1],
            [8, 4, 3, 1, 0],
            [5, 4, 3, 2, 1]])

#--------------TRAINING OUTPUT--------------#
y = np.array([[1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]])

#--------------ALPHA TUNING--------------#
#Put in a fair range, usually powers of ten I think are traditional.
#If you know what alpha you want just put that one in.
alphas = [0.01, 0.1, 1, 10]


#--------------TESTING--------------#
test_x = np.array([[2, 2, 2, 2, 2],
                    [1, 2, 5, 6, 9],
                    [7, 3, 2, 1, 0],
                    [2, 3, 7, 9, 10],
                    [1, 4, 1, 4, 1],
                    [1, 2, 4, 2, 5],
                    [1, 5, 3, 12, 0],
                    [1, 1, 1, 1, 2]])

expected_out = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0], # Something I wanted to try, kinda weird
                            [1, 0, 0], # Also weird, but makes sense
                            [1, 0, 0], # Same as above
                            [0, 1, 0]])# Interesting


run_network(x, y, num_layers=4, synapse_size=4, alpha=alphas,
                num_iter=300000, test_input=test_x, test_output=expected_out)
