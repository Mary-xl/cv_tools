# cv_tools
This project is built as a toolset for computer vision studies. It includes basic image processing operations (based on OpenCV) and will be extended in the future.
It is also used as an practice for a Deep Learning training course.

1. Dependences:
   
   OpenCV (v3 and above)
   Python (v3 and above), numpy, matplotlib, docopt..

2. How to run:
   (1) Basic image processing:
       Usage:
       run_pipeline.py <dataFile>
                     (--local | --server)
                     [--basic (--crop | --shift | --gamma | --transform |--hist_equa)]
       e.g.
       python run_pipeline.py  /home/mary/AI_Computing/CV_DL/data/Lenna.png --local --basic --gamma
      
