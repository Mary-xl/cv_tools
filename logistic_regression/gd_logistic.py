
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.datasets as data

def sigmoid (z):
    hx = 1.0/(1+np.exp(-z))
    return hx

def gradient_function(X,y,theta,m):
    z = np.dot(X, theta)
    hx=sigmoid(z)
    diff=hx-y
    gradient=1./m*np.dot(np.transpose(X),diff)

    return gradient

def loss_function(X,y,theta,m):
    z = np.dot(X, theta)
    hx=sigmoid(z)
    J=-(y*np.log(hx)+(1-y)*np.log(1-hx)).mean()
    return J

def gradient_descent(X,y,theta,lr,m):
    gradient=gradient_function(X,y,theta,m)
    theta=theta-lr*gradient
    J=loss_function(X,y,theta,m)
    print ('theta: {0}'.format(theta))
    print('loss: {0}'.format(J))
    return  theta,J

def predict_prob(gz):
    return gz>=0.5

def load_data():

    iris = data.load_iris()
    X1 = iris.data[:, :2]
    X0=np.ones((len(X1),1))
    X=np.hstack((X0,X1))
    y = (iris.target != 0) * 1
    return X,y

def train(sample_X, sample_y, batch_size,lr, maxIrr):
    theta = np.zeros(sample_X.shape[1])
    total_size=len(sample_X)
    for i in range (maxIrr):
        batch_idx=np.random.choice(total_size,batch_size)
        batch_X=sample_X[batch_idx,:]
        batch_y=sample_y[batch_idx]
        theta,J=gradient_descent(batch_X, batch_y, theta, lr, batch_size)

    return theta,J


def run( batch_size,lr,maxIrr):
    X,y=load_data()
    theta,J=train(X,y,batch_size,lr,maxIrr)

    print('===================================')
    print ('estimated theta: {0}'.format(theta))
    print('final loss: {0}'.format(J))
    return theta

def visualise (theta):
    X, y = load_data()
    X2_estimated=-(theta[0]+theta[1]*X[:,1])/theta[2]
    fig, ax = plt.subplots()

    ax.scatter(X[:,1],X[:,2],c=y, alpha=0.8, edgecolors='none')
    ax.plot(X[:, 1],X2_estimated, color='blue')
    plt.show()

if __name__=='__main__':

    batch_size=50
    lr=0.01
    maxIrr=30000
    theta=run(batch_size,lr,maxIrr)
    visualise(theta)