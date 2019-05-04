from imutils import face_utils

import os, sys
import argparse
import imutils
import dlib
import cv2

ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='a crop utility')

# Argument are :-- shape_predictor_68_face_landmarks.dat
parser.add_argument('-p', '--shape-predictor', type=str, nargs='?',
    help='path to facial landmark predictor')

parser.add_argument('--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')

FLAGS = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FLAGS.shape_predictor)

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/"]

def resize( path ):
    items = os.listdir(path)

    for filename in items:

        if (filename.endswith('.mp4')): #or .avi, .mpeg, whatever. 

            # outdir = os.path.join( path, filename+"_dir" )
            # os.makedirs( outdir )
            # os.system( "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path, filename ), os.path.join( outdir, "output%d.jpeg" ) ) )

            os.system( "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path, filename ), os.path.join( path, filename+"%d.jpeg" )))     

            items1 = os.listdir(path)
            # with os.system(os.path.join( outdir, open(filename+"_dir.csv", 'wb' ))) as file:
            with open(os.path.join( path+"/out", filename+"_dir.csv"), 'wb' ) as file:
                for item in items1:

                    if item == '.DS_Store':
                       continue

                    if (item.endswith('.jpeg')):
                      
                        line = "\""+item+"\";"

                        # load the input image, resize it, and convert it to grayscale
                        images = cv2.imread(path+item)

                        images = imutils.resize(images, width=500)
                        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
                        f, e = os.path.splitext(path+item)

                        # Remove item into dir
                        os.remove(path+item)

                        rects = detector(gray, 1)

                        for (i, rect) in enumerate(rects):

                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            for (x, y) in shape:
                                    
                                line = line + ";\""+str(x)+"-"+str(y)+"\""

                            file.write(line.encode())
                            file.write('\n'.encode()) 
        else:
          
             for root, dirs, files in os.walk(path+filename, topdown=False):

                 for name in files:
                     
                     if (name.endswith(('.mp4','.avi'))): #or .avi, .mpeg, whatever.

                         # resA = os.popen("ffmpeg -i " + os.path.join( path+filename, name )+ " -f null -").read()
                         
                         # print("START-----"os.popen("ffmpeg -t -frames"+os.path.join( path+filename, name )).read(),"-----STOP")
                         os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path+filename, name ), os.path.join(path+filename, name+"%d.jpeg" )))     

                         items1 = os.listdir(root+"/")

                         with open(os.path.join( path+"/out", name+"_dir.csv"), 'wb' ) as file:
                              for item in items1:

                                  if item == '.DS_Store':
                                     continue

                                  if (item.endswith('.jpeg')):
                                    
                                      line = "\""+item+"\";"

                                      # load the input image, resize it, and convert it to grayscale
                                      images = cv2.imread(root+"/"+item)

                                      # print("Hellheyyyy------",images)

                                      images = imutils.resize(images, width=500)
                                      gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
                                      f, e = os.path.splitext(root+"/"+item)

                                      # Remove item into dir
                                      os.remove(root+"/"+item)

                                      rects = detector(gray, 1)

                                      for (i, rect) in enumerate(rects):

                                          # determine the facial landmarks for the face region, then
                                          # convert the facial landmark (x, y)-coordinates to a NumPy
                                          # array
                                          shape = predictor(gray, rect)
                                          shape = face_utils.shape_to_np(shape)

                                          for (x, y) in shape:
                                                  
                                              line = line + ";\""+str(x)+"-"+str(y)+"\""

                                          file.write(line.encode())
                                          file.write('\n'.encode()) 
                     else:
                         continue                     
               
for path in paths:
    resize( path )
