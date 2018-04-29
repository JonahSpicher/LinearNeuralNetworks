import numpy as np

#sigmoid
def nonlin(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

#input
x = np.array([[1, 1, 0],
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1]])

#correct output
y = np.array([[0, 1],
            [0, 1],
            [1, 0],
            [1, 0]]).T


np.random.seed(1)

syn0 = 2*np.random.random((3, 2))-1

for i in range(40000):
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    l1_error = l1 - y
    if i%1000 == 0:
        print("error:\n", l1_error)

    l1_delta = l1_error*nonlin(l1, True)
    syn0_deriv = np.dot(l0.T, l1_delta)

    syn0 -= np.dot(l0.T, l1_delta)
print("Output After Training:")
print(l1)
print("Synapse:\n", syn0)


# test_x = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
#
# expected_y = np.array([[0, 0, 1, 1]]).T
#
# test_data = nonlin(np.dot(test_x, syn0))
#
# print("test:\n", test_data)
