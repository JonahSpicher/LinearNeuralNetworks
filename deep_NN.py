import numpy as np

#Basically, making these numbers bigger should make it smarter but also take longer.
DEPTH = 4
SYN_SIZE = 4

#--------------FUNCTIONS--------------#
#"sigmoid," takes a number and turns it into a decimal
def nonlin(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


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

#--------------INITIALIZING--------------#
syns = [None]*(DEPTH-1)
layers = [None]*(DEPTH)
errors = [None]*(DEPTH-1)
deltas = [None]*(DEPTH-1)

#Put in a fair range, usually powers of ten I think are traditional. If you know what alpha you want just put that one in.
alphas = [0.01, 0.1, 1]





#--------------TRAINING--------------#
for alpha in alphas:
    print("Testing alpha value of %s" % (alpha))

    np.random.seed(1)
    if DEPTH==2:
        syns[0] = 2*np.random.random((len(x[1]), 1))-1
    else:
        syns[0] = 2*np.random.random((len(x[1]), SYN_SIZE))-1
    for i in range(1, DEPTH-1):
        if i == DEPTH-2:
            syns[i] = 2*np.random.random((SYN_SIZE, len(y[0])))-1
        else:
            syns[i] = 2*np.random.random((SYN_SIZE, SYN_SIZE))-1

    for i in range(300000):
        #set first layer equal to input
        layers[0] = x
        #Make other layers
        for j in range(1, DEPTH):
            layers[j] = nonlin(np.dot(layers[j-1], syns[j-1]))

        #Find error
        errors[-1] = layers[-1] - y
        deltas[-1] = errors[-1]*nonlin(layers[-1], True)

        #if i%100000 == 0:
            #print("error:\n", errors[-1])

        #backpropogate to find other errors
        for j in range(2, DEPTH):
            errors[-j] = deltas[-j+1].dot(syns[-j+1].T)
            deltas[-j] = errors[-j]*nonlin(layers[-j], True)

        if DEPTH==2:
            syns[0] -= alpha*(layers[0].T.dot(deltas[0]))
        else:
            for j in range(DEPTH-1):
                syns[j] -= alpha*(layers[j].T.dot(deltas[j]))

    print("Output After Training:")

    answer = layers[-1]

# #If you want it to be pretty
# for j in range(len(answer)):
#     for i in range(len(answer[0])):
#         if answer[j][i]>0.9:
#             answer[j][i] = 1
#         elif answer[j][i]<0.1:
#             answer[j][i] = 0
    print(answer)


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
                            [0, 1, 0], # Test, kinda weird
                            [1, 0, 0], # Also Test, but makes sense
                            [1, 0, 0], # Same as above
                            [0, 1, 0]])# Interesting
#set first layer equal to input
    layers[0] = test_x
#Make other layers
    for j in range(1, DEPTH):
        layers[j] = nonlin(np.dot(layers[j-1], syns[j-1]))

        answer = layers[-1]
# #If you want it to be pretty
# for j in range(len(answer)):
#     for i in range(len(answer[0])):
#         if answer[j][i]>0.9:
#             answer[j][i] = 1
#         elif answer[j][i]<0.1:
#             answer[j][i] = 0
    print("Test:\n", answer)
    test_error = expected_out - answer
    print("Final Error:\n", test_error)
