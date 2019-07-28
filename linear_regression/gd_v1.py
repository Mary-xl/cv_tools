
import numpy as np
import random

def inference (w,b,x):
    pred_y=w*x+b
    return pred_y

def gradient (pred_y,gt_y,x):
    dw=(pred_y-gt_y)*x
    db=pred_y-gt_y
    return  dw, db

def theta_step_update (batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw=0
    avg_db=0
    batch_size=len(batch_x_list)

    for i in range (batch_size):
        pred_y=inference(w,b,batch_x_list[i])
        dw,db=gradient(pred_y,batch_gt_y_list[i],batch_x_list[i])
        avg_dw+= dw
        avg_db+= db
    avg_dw/=batch_size
    avg_db/=batch_size

    #update simultanously
    w=w-lr*avg_dw
    b=b-lr*avg_db

    return w,b

def loss_function (batch_x_list, batch_gt_y_list, w, b):
    batch_size=len(batch_x_list)
    sum_loss=0
    for i in range (batch_size):
        single_loss=(w*batch_x_list[i]+b-batch_gt_y_list[i])**2
        sum_loss+=single_loss
    J=sum_loss/2*batch_size
    return J

def train(x_list, gt_y_list, batch_size,lr, maxIrr):
    #initial setup: pick up a starting point for (w,b);
    w=0
    b=0
    total_size=len(x_list)

    # randomly choose batch size of samples from the whole data
    # throw the batch into the iteration as training data for each round
    for i in range (maxIrr):
        batch_idx=np.random.choice(total_size,batch_size)
        batch_x_list=[x_list[j] for j in batch_idx]
        batch_gt_y_list=[gt_y_list[j] for j in batch_idx]
        w,b=theta_step_update(batch_x_list,batch_gt_y_list,w,b,lr)
        J=loss_function(batch_x_list,batch_gt_y_list,w,b)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss: {0}'.format(J))
    return w,b,J

def generate_data(num_samples):
    w=random.randint(0,10)+random.random()
    b=random.randint(0,5)+random.random()
    x_list=[]
    gt_y_list=[]
    for i in range (num_samples):
        x = random.randint(0, 100) * random.random()
        y=w*x+b+random.random()* random.randint(-1, 1)
        x_list.append(x)
        gt_y_list.append(y)
    return w,b,x_list,gt_y_list

def run(num_samples, batch_size,lr,maxIrr):
    w0,b0,x_list,gt_y_list=generate_data(num_samples)
    w,b,J=train(x_list, gt_y_list, batch_size,lr, maxIrr)
    print ('original w: {0}, b:{1}'.format(w0,b0))
    print ('estimated w: {0}, b:{1}'.format(w,b))
    print ('final loss J: {0}'.format(J))

if __name__=='__main__':
    num_samples=100
    batch_size=50
    lr=0.001
    maxIrr=10000
    run(num_samples, batch_size,lr,maxIrr)


