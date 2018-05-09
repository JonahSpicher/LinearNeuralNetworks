import numpy as np

#"sigmoid," takes a number and turns it into a decimal
def nonlin(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


#training input
x = np.array([[0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 1]])

#correct output
y = np.array([[1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]])

#this is good to do
np.random.seed(1)




syn0 = 2*np.random.random((4, 5))-1
syn1 = 2*np.random.random((5, 5))-1
syn2 = 2*np.random.random((5, 5))-1
syn3 = 2*np.random.random((5, 2))-1

for i in range(300000):
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    l4 = nonlin(np.dot(l3, syn3))

    l4_error = y - l4
    l4_delta = l4_error*nonlin(l4, True)

    if i%10000 == 0:
        print("error:\n", l4_error)

    l3_error = l4_delta.dot(syn3.T)
    l3_delta = l3_error*nonlin(l3, True)

    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error*nonlin(l2, True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, True)

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)
    syn3 += l3.T.dot(l4_delta)
print("Output After Training:")

answer = l4
# for i in range(len(answer)):
#     if answer[i]>0.9:
#         answer[i] = 1
#     elif answer[i]<0.1:
#         answer[i] = 0
print(answer)

new_x = np.array([[0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0]])
l0 = new_x
l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))
l3 = nonlin(np.dot(l2, syn2))
l4 = nonlin(np.dot(l3, syn3))
print("Test:\n", l4)
