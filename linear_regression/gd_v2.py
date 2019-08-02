
import numpy as np
import random
import matplotlib.pyplot as plt

def gradient_function(X,y,theta,m):
    diff=np.dot(X,theta)-y
    step=(1./m)*np.dot(np.transpose(X),diff)
    return step

def loss_function(X,y,theta,m):
    diff=np.dot(X,theta)-y
    J=(1./2*m)*np.dot(np.transpose(diff),diff)
    return J

def gradient_descent(X,y,theta,lr, m):
    theta=theta-lr*gradient_function(X,y,theta,m)
    J=loss_function(X,y,theta,m)
    print ('theta: {0}'.format(theta))
    print('loss: {0}'.format(J))
    return  theta,J

def generate_data(num_samples):
    theta0=(random.randint(0,10)+random.random(),random.randint(0,5)+random.random())
    X1=[]
    for i in range (num_samples):
        x = random.randint(0, 100) * random.random()
        X1.append(x)
    X1=np.reshape(X1,(num_samples,1))
    X0=np.ones((num_samples,1))
    X=np.hstack((X1,X0))

    y=np.dot(X,theta0)
    y=np.reshape(y,(num_samples,1))
    rand=random.random()*np.random.randint(-1,1,size=(num_samples, 1))
    y=y+rand
    return X,y,theta0

def train(sample_X, sample_y, batch_size,lr, maxIrr):
    theta = np.array([0, 0]).reshape(2, 1)
    total_size=len(sample_X)
    for i in range (maxIrr):
        batch_idx=np.random.choice(total_size,batch_size)
        batch_X=sample_X[batch_idx,:]
        batch_y=sample_y[batch_idx,:]
        theta,J=gradient_descent(batch_X, batch_y, theta, lr, batch_size)

    return theta,J

def run(num_samples, batch_size,lr,maxIrr):
    X,y,theta0=generate_data(num_samples)
    theta,J=train(X,y,batch_size,lr,maxIrr)
    fig, ax = plt.subplots()
    ax.scatter(X[:,0],y,c='blue', alpha=0.3, edgecolors='none')
    ax.plot(X[:,0],np.dot(X,theta0), color='red')
    ax.plot(X[:,0],np.dot(X,theta),color='green')
    plt.show()

    print('===================================')
    print ('original theta: {0}'.format(theta0))
    print ('estimated theta: {0}'.format(theta))
    print('final loss: {0}'.format(J))


if __name__=='__main__':
    num_samples=100
    batch_size=50
    lr=0.001
    maxIrr=30000
    run(num_samples, batch_size,lr,maxIrr)