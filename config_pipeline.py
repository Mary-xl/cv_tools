
"""
Created on July 05 2019

@author: Xun Li, xunlimary@hotmail.com

"""
import os

class config():



   def __init__(self, dataFile,RunMode):

       if RunMode=='local':
          print ('local environment')
       if RunMode=='server':
          print ('server environment')

       self.root_path=os.path.dirname(os.path.abspath(__file__))
       self.dataFile=dataFile