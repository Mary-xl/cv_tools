

"""
Created on 05 July 2019
@author: Mary LI, xunlimary@hotmail.com

"""


from docopt import docopt
import cv2
from config_pipeline import config
from opencv_basics.colour_shift import shift_color
from opencv_basics.adjust_gamma import gamma_correction
from opencv_basics.transform import transform_img
from opencv_basics.histogram_equalization import histogram_equalization


CLI_OPTS = """
Usage:
     run_pipeline.py <dataFile>
                     (--local | --server)
                     [--basic (--shift | --gamma | --transform |--hist_equa)]
                     


Options:
     --local                          Run on local processing environment
     --server                         Run on server
     --basic                          basic image processing using OpenCV
     --shift                          shift intensities in all 3 channels                    
     --gamma                          perform gamma correction                     
     --transform                      perform image transformation, including similarity, affine and perspective transformations                       
     --hist_equa                      perform histogram equalization                         
                                
"""

opts = docopt(CLI_OPTS)
# Default values
RunMode = 'local'
basic = False
shift = False
gamma = False
resize = False
hist_equa=False
transform = False



try:
    dataFile=opts['<dataFile>']
except Exception as e:
    print("Please specify data file")
    raise e

if opts['--local']:
    RunMode = 'local'
if opts['--server']:
    RunMode = 'server'
if opts['--basic']:
    basic = True
if opts['--shift']:
    shift = True
if opts['--gamma']:
    gamma = True
if opts['--hist_equa']:
    hist_equa = True
if opts['--transform']:
    transform = True


def main():

    c=config(dataFile,RunMode)

    if basic==True:
        img = cv2.imread(c.dataFile)
        dtype = img.dtype
        (w, h, d) = img.shape
        img_dim = img.shape
        img_center = (w // 2, h // 2)


        if shift==True:
           print (">>>>>shift colours for RGB image<<<<<")
           s=int(input("please input a shift value in range [0,255]:"))
           shift_img=shift_color(img, s)

        elif gamma==True:
           print (">>>>>perform gamma correction<<<<<")
           G=float(input("please input a gamma value:"))
           gamma_img=gamma_correction(img,G)

        elif transform==True:
            print (">>>>>perform transform<<<<<")
            transform_img(img,img_center,img_dim)

        elif hist_equa==True:
            print (">>>>>perform histogram equalization<<<<<")
            histogram_equalization(img)


if __name__ == '__main__':
    main()