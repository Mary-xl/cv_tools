
from __future__ import print_function
import torch
from torch.autograd import Variable


def test1():
   x=torch.empty(5,3)
   print (x)

   y = torch.rand(5, 3)
   print(y)

   z = torch.zeros(5, 3, dtype=torch.long)
   print(z)

   result = torch.empty(5, 3)
   torch.add(x,y, out=result)
   print (result)

def testGradient():

    x = torch.tensor([2.0, 1.0],requires_grad=True)
    print(x)
    print('\n')

    y=2*x+1
    print (y)
    print (y.grad_fn)
    print('\n')

    z=y*y*3
    print (z)
    print (z.grad_fn)
    print ('\n')

    out=z.mean()
    print (out)
    print (out.grad_fn)

    out.backward()
    print (z.grad)
    print (y.grad)
    print (x.grad)



if __name__ == '__main__':
    testGradient()