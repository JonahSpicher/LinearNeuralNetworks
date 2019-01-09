"""Simple neural network, trains to classify an input array.
Depth and synapse size are parameterized.
"""
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

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
    norm = np.linalg.norm(error)
    cost = 0.5*(norm**2)
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

def run_once(x, syns):
    """Given an input and a list of synapses, calculates an output. Can be used
    for prediction or for training.
    """
    layers = []
    layers.append(x)
    for j in range(len(syns)):
        # print(j)
        # print("Layers:", len(layers))
        # print("syns:", len(syns))
        layers.append(nonlin(np.dot(layers[j], syns[j])))
    return layers

def update_syns(y, layers, alpha, syns, num_layers):
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
            update = alpha*(layers[j].T.dot(deltas[j]))
            syns[j] -= update

    return syns, errors


def train_one_alpha(x, y, num_layers, synapse_size, alpha, num_iter):
    """Trains the neural network once, for one alpha value.
    """
    syns = make_syns(x, y, num_layers, synapse_size)
    for i in range(num_iter):
        layers = run_once(x, syns)
        syns, errors = update_syns(y, layers, alpha, syns, num_layers)
    #print("Output After Training:\n", layers[-1].shape)
    output_error = errors[-1]
    #print("Final Error:\n", output_error)
    final_cost = cost(output_error)
    #print("Final Cost:\n", final_cost)
    return layers, syns, output_error, final_cost

def run_network(x, y, num_layers, synapse_size, alphas=[1],
                num_iter=300000, test_info=(0,)):
    """Trains weights for several different alpha values. Stores the cost score
    for each alpha and reports the lowest cost when finished. Can optionally take
    test_info. If provided, will also find test cost and report least cost.

    Inputs:
    x: two dimensional numpy array, the training input.

    y: two dimensional numpy array, the training solution set or key.

    num_layers: integer, sets the number of hidden layers. Includes output and input, so
    setting num_layers=2 would create 0 hidden layers.

    synapse_size: integer, sets the side length of the hidden layers, when possible. First
    and last hidden layers will have one side constrained, interior hidden layers
    will be square. For a 3 layer network (the most common), a good rule of thumb is to
    set this equal to the mean of the length of a single input and the length of a single output.

    alphas: list, a set of alpha values to tune step size for optimization. By
    default, no alpha tuning occurs.

    num_iter: integer, the number of times the network will backpropagate. Keep in mind,
    this many iterations will happen for each alpha value. Default is 300000.

    test_info: tuple, an optional setting which cause sthe network to use
    generated weights on a test input and compare to an expected output. By default,
    no info is provided, and this step will be skipped. To use, set test_info
    equal to a tuple containing the test input array and the expected output
    array, as in test_info=(input, output).


    """
    least_training_cost=100000
    training_tuple = (least_training_cost, None)
    test_flag = False
    syn_dict = {}
    if len(test_info) != 1:
        test_flag = True
        least_test_cost=100000
        test_tuple = (least_test_cost, None)




    for alpha in alphas:
        print("Testing alpha value of %s" % (alpha))
        layers, syns, output_error, training_cost = train_one_alpha(x, y, num_layers, synapse_size, alpha, num_iter)
        syn_dict[alpha] = syns
        print("Cost:", training_cost)
        if test_flag == True:
            test_layers = run_once(test_info[0], syns, num_layers)
            output = test_layers[-1]
            test_error = test_info[1]-output
            test_cost = cost(test_error)

            if test_cost < least_test_cost:
                least_test_cost = test_cost
                test_tuple = (test_cost, alpha)


        if training_cost < least_training_cost:
            least_training_cost = training_cost
            training_tuple = (training_cost, alpha)

    if test_flag == True:
        print("Least costs:")
        print("For training:\n", training_tuple)
        print("For test:\n", test_tuple)

    else:
        print("Least Training Cost:\n", training_tuple)
        np.save("weights", syn_dict[training_tuple[1]])



def predict(x, syns):
    """Predicts the class of a given input.
    """
    layers = run_once(x, syns)
    classes = layers[-1]
    best = 0
    for i in range(len(classes)):
        if classes[i] > best:
            best = classes[i]
            prediction = i
    return prediction



def get_digits():
    """Gets scikit learn's handwritten image database and loads them in a form the
    network will take."""
    data, target = load_digits(return_X_y=True)
    #print(len(data), len(target))
    training = []
    train_key = np.zeros((len(data), 10))
    for i in range(len(data)):
        training.append(data[i])
        train_key[i][target[i]] = 1
    training = np.array(training)
    print(training.shape)
    return training, train_key


    # print(digits.DESCR)
    # fig = plt.figure()
    # for i in range(10):
    #     subplot = fig.add_subplot(5, 2, i + 1)
    #     subplot.matshow(np.reshape(digits.data[i], (8, 8)), cmap='gray')
    # plt.show()

def add_bias(data):
    """
    Adds bias terms, but only to the first layer. Provide the training input dataset
    """
    new_data = list(data)
    for i in range(len(new_data)):
        new_data[i] = np.append(new_data[i], 1)
    return np.array(new_data)





#--------------TRAINING INPUT TEMPLATE--------------#
x = np.array([[1, 2, 3, 4, 5],
            [0, 3, 4, 5, 6],
            [2, 4, 6, 7, 8],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4],
            [7, 6, 3, 2, 1],
            [8, 4, 3, 1, 0],
            [5, 4, 3, 2, 1]])

#--------------TRAINING OUTPUT TEMPLATE--------------#
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
alphas = [0.001]

#--------------TESTING TEMPLATES--------------#
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

if __name__ == "__main__":
    x, y = get_digits()
    x = add_bias(x)
    run_network(x, y, num_layers=4, synapse_size=37, alphas=alphas,
                num_iter=500000)
