
import numpy as np
import random
import matplotlib.pyplot as plt
from operator import itemgetter


def gradient_function(X,y,theta,m):
    diff=np.dot(X,theta)-y
    gradient=(1./m)*np.dot(np.transpose(X),diff)
    return gradient

def loss_function(X,y,theta,m):
    diff=np.dot(X,theta)-y
    J=(1./2*m)*np.dot(np.transpose(diff),diff)
    return J

def gradient_descent(X,y,theta,lr, m):
    gradient=gradient_function(X,y,theta,m)
    theta=theta-lr*gradient
    J=loss_function(X,y,theta,m)
    print ('theta: {0}'.format(theta))
    print('loss: {0}'.format(J))
    return  theta,J,gradient

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
    result=[]
    for i in range (maxIrr):
        batch_idx=np.random.choice(total_size,batch_size)
        batch_X=sample_X[batch_idx,:]
        batch_y=sample_y[batch_idx,:]
        theta,J,gradient=gradient_descent(batch_X, batch_y, theta, lr, batch_size)
        flat_res = np.hstack((i,gradient.flatten(), theta.flatten(),J.ravel()))
        if  i==0:
            result=flat_res
        else:
            result=np.vstack((result,flat_res))

    fig0=plt.figure()
    ax0=fig0.add_subplot()
    ax0.scatter(result[9:,0],result[9:,5],c='yellow',s=1,label='batch_size: '+str(batch_size))  #J
    ax0.set_ylabel('J_loss')
    ax0.set_xlabel('iterations with batch size: '+str(batch_size))

    fig1 = plt.figure()
    ax1 =fig1.add_subplot(2,1,1)
    ax1.scatter(result[5:,0],result[5:,1],c='blue',s=1, label='batch_size: '+str(batch_size))
    ax1.set_ylabel('gradient_theta1')
    ax1.set_ylim([-30, 30])
    ax1.set_xlabel('iterations')
    ax2 =fig1.add_subplot(2,1,2)
    ax2.scatter(result[5:,0],result[5:,2],c='green',s=1,label='batch_size: '+str(batch_size))
    ax2.set_ylabel('gradient_theta2')
    ax2.set_xlabel('iterations with batch size: '+str(batch_size))


    fig2 = plt.figure()
    ax3 =fig2.add_subplot(2,1,1)
    ax3.scatter(result[5:,0],result[5:,3],c='blue',s=1, label='batch_size: '+str(batch_size))
    ax3.set_ylabel('theta1')
    ax3.set_xlabel('iterations')
    ax4 =fig2.add_subplot(2,1,2)
    ax4.scatter(result[5:,0],result[5:,4],c='green',s=1,label='batch_size: '+str(batch_size))
    ax4.set_ylabel('theta2')
    ax4.set_xlabel('iterations with batch size: '+str(batch_size))
    plt.show()

    return theta,J

def run(num_samples, batch_size,lr,maxIrr):
    X,y,theta0=generate_data(num_samples)
    theta,J=train(X,y,batch_size,lr,maxIrr)
    # fig, ax = plt.subplots()
    # ax.scatter(X[:,0],y,c='blue', alpha=0.3, edgecolors='none')
    # ax.plot(X[:,0],np.dot(X,theta0), color='red')
    # ax.plot(X[:,0],np.dot(X,theta),color='green')
    # plt.show()

    print('===================================')
    print ('original theta: {0}'.format(theta0))
    print ('estimated theta: {0}'.format(theta))
    print('final loss: {0}'.format(J))


if __name__=='__main__':
    num_samples=500
    batch_size=300
    lr=0.001
    maxIrr=30000
    run(num_samples, batch_size,lr,maxIrr)