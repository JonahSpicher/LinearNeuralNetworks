import numpy as np
import matplotlib.pyplot as plt

#sigmoid
def nonlin(x, deriv=False):
    """Sigmoid Function, set deriv=True to get the derivative of the sigmoid.
    """
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def abline(slope, intercept, graph, scale):
    """Plot a line from slope and intercept"""
    x_vals = np.linspace(-scale, scale)
    y_vals = intercept + slope * x_vals
    graph.plot(x_vals, y_vals, '--')

#input
x = np.array([[1, 2],
            [4, 7],
            [6, -3],
            [2, -2],
            [-6, 2],
            [-4, 4],
            [-2, -1],
            [-2, -3]])

#correct output
y = np.array([[0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1]])


#Generate a random synapse with a mean value of 0
np.random.seed(1)
syn0 = 2*np.random.random((len(x[0]), len(y[0])))-1
print("Initial synapse:\n", syn0)

for i in range(100000):
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    l1_error = l1 - y
    if i%10000 == 0:
        print("error:\n", l1_error)

    l1_delta = l1_error*nonlin(l1, True)
    syn0_deriv = np.dot(l0.T, l1_delta)

    syn0 -= np.dot(l0.T, l1_delta)
print("Output After Training:\n")
print(l1, '\n')

#Rounding
answer = l1
for j in range(len(answer)):
    for i in range(len(answer[0])):
        if answer[j][i]>0.9:
            answer[j][i] = 1
        elif answer[j][i]<0.1:
            answer[j][i] = 0
print("Rounded:\n", answer)
print("Final Error:\n")
print(l1_error)


#Decision boundary info
print("End synapse:\n", syn0)
plot_points_x = []
plot_points_y = []
for i in range(len(x)):
    result = np.dot(x[i], syn0)
    plot_points_x.append(result[0])
    plot_points_y.append(result[1])
    print("Result for %s is %s"%(x[i], result))



X, Y = zip(*syn0)
plt.figure()
ax = plt.gca()
#Find slopes of the vectors
slope_v1 = Y[0]/X[0]
slope_v2 = Y[1]/X[1]
#perpendicular line slopes
decision_bound1_slope = -1/slope_v1
decision_bound2_slope = -1/slope_v2
#Makes weight vectors
ax.quiver(X[0], Y[0], angles='xy', scale_units='xy', scale=1)
ax.quiver(X[1], Y[1], angles='xy', scale_units='xy', scale=1)
#Decision boundaries
abline(decision_bound1_slope, 0, ax, scale=3)#scale should be adjusted to make a nicely scaled graph for each problem
abline(decision_bound2_slope, 0, ax, scale=30)
ax.plot(plot_points_x, plot_points_y, 'ro')


plt.draw()
plt.show()
# print("Synapse:\n", syn0)
