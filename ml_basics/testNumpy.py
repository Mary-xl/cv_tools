
import numpy as np
import torch
import torch.nn as nn

def usingNumpy():
   N, D_in, H, D_out=64,1000,100,10
   x=np.random.randn(N,D_in)
   y=np.random.randn(N,D_out)
   w1=np.random.randn(D_in,H)
   w2=np.random.randn(H,D_out)

   lr=1e-6

   for i in range(500):
      #forward
      z1=x.dot(w1)
      a1=np.maximum(z1,0) #RELU
      y_pred=a1.dot(w2)
      loss=np.square(y_pred-y).sum

      #backward
      grad_y_pred=2*(y_pred-y)
      grad_w2=a1.T.dot(grad_y_pred)
      grad_a1=grad_y_pred.dot(w2.T)
      grad_z1=grad_a1.copy
      grad_z1[grad_z1<0]=0
      grad_w1=x.T.dot(grad_z1)

      #update weights
      w1-=lr*grad_w1
      w2-=lr*grad_w2

      print ("i, loss: ",i, "  ",loss)

def usingNN():
   N, D_in, H, D_out = 64, 1000, 100, 10
   x=torch.randn(N,D_in)
   y=torch.randn(N, D_out)

   model=nn.Sequential(
      nn.Linear(D_in,H),
      nn.ReLU(),
      nn.Linear(H,D_out)
   )

   loss_fn=nn.MSELoss(reduction='sum')

   lr=1e-4
   optimizer=torch.optim.Adam(model.parameters(),lr=lr)

   for i in range(500):
      y_pred=model(x)
      loss=loss_fn(y_pred,y)
      if i%100==99:
         print (i, 'loss: ', loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()




if __name__=='__main__':
    #usingNumpy()
    usingNN()