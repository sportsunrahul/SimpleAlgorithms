import numpy as np
import tensorflow as tf
e = 1e-12

def sigmoid(x):
    return 1/(1+np.exp(-x))

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(np.dot(theta, x.T))

def cost_function(x,y, theta):
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)+e) + (1 - y) * np.log(
            1 - probability(theta, x)+e))
    return total_cost

def grad_wrt_theta(theta, x, y):
    m = x.shape[0]
    return (1/m) * np.dot(sigmoid(np.dot(theta,x.T)) - y, x)

def logistic_regression(x,y):
    theta = np.random.normal(0,1,(1, num_features))
    lr = 0.1
    print("Before Training: ")
    p = probability(theta, x[y==0])
    print("Accuracy on digit 0: ", len(p[p<0.5])/len(x[y==0]))
    p = probability(theta, x[y==1])
    print("Accuracy on digit 1: ", len(p[p>0.5])/len(x[y==1]))

    for i in range(50):
        cost = cost_function(x,y, theta)
        theta = theta - lr * grad_wrt_theta(theta,x,y)

    return theta

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

x_train, x_test = x_train.reshape((x_train.shape[0],-1)), x_test.reshape((x_test.shape[0],-1))
num_samples, num_features = x_train.shape
num_class = 10

index_0 = np.where(y_train == 0)
x_train_0 = x_train[index_0]
index_1 = np.where(y_train == 1)
x_train_1 = x_train[index_1]
x_train = np.vstack((x_train_0,x_train_1))
y_train = np.hstack((np.zeros(len(x_train_0)), np.ones(len(x_train_1))))



theta = logistic_regression(x_train,y_train)

print("After Training: ")
p = probability(theta, x_train_0)
print("Accuracy on digit 0: ", len(p[p<0.5])/len(x_train_0))
p = probability(theta, x_train_1)
print("Accuracy on digit 1: ", len(p[p>0.5])/len(x_train_1))










