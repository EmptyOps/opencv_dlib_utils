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

ext = (".avi", ".mp4")

# Argument are :-- shape_predictor_68_face_landmarks.dat
parser.add_argument('-p', '--shape-predictor', type=str, nargs='?',
    help='path to facial landmark predictor')

parser.add_argument('-d', '--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')
parser.add_argument('-ik', '--is_keep_extracted_image', action='store_true', help='A boolean True False')

FLAGS = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FLAGS.shape_predictor)

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/"]

def toint(str):
  try:
    return int(str)
  except Exception as e:
    return 0
  else:
    pass
  finally:
    pass

def resize( path ):
    items = os.listdir(path)

    for filename in items:

        if (filename.endswith(ext)): #or .avi, .mpeg, whatever. 

            # outdir = os.path.join( path, filename+"_dir" )
            # os.makedirs( outdir )
            # os.system( "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path, filename ), os.path.join( path, "output%d.jpeg" ) ) )

            #single image only for testing "-vframes 50"
            os.system( "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path, filename ), os.path.join( path, os.path.join( "images", "%d.jpeg" ) ) ) )     

            items1 = os.listdir( os.path.join( path, "images" ) ) 
            items1 = sorted(items1,key=lambda x: toint(os.path.splitext(x)[0]))

            # with os.system(os.path.join( outdir, open(filename+"_dir.csv", 'wb' ))) as file:
            with open(os.path.join( path+"/out", filename+"_dir.csv"), 'wb' ) as file:

                for item in items1:

                    if item == '.DS_Store':
                       continue

                    if (item.endswith('.jpeg')):
                      
                        # load the input image, resize it, and convert it to grayscale
                        images = cv2.imread( os.path.join( path, "images", item ) ) 

                        images = imutils.resize(images, width=500)
                        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
                        f, e = os.path.splitext( os.path.join( path, "images", item ) )

                        # Remove item into dir
                        if not FLAGS.is_keep_extracted_image:
                          os.remove(os.path.join( path, "images", item ) )

                        rects = detector(gray, 1)

                        for (i, rect) in enumerate(rects):

                            line = "\""+item+"~"+str(i)+"\";"

                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            for (x, y) in shape:
                                    
                                line = line + ";\""+str(x)+"~"+str(y)+"\""

                            file.write(line.encode())
                            file.write('\n'.encode()) 
        else:
          
              for root, dirs, files in os.walk(path+filename, topdown=False):

                  for name in files:
                     
                      if (name.endswith(ext)): #or .avi, .mpeg, whatever.

                         os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path+filename, name ), os.path.join(path+filename, name+"%d.jpeg" )))     

                         items1 = os.listdir(root+"/")

                         with open(os.path.join( path+"/out", name+"_dir.csv"), 'wb' ) as file:

                              for item in items1:

                                  if item == '.DS_Store':
                                     continue

                                  if (item.endswith('.jpeg')):

                                      # load the input image, resize it, and convert it to grayscale
                                      images = cv2.imread(root+"/"+item)

                                      # print("Hellheyyyy------",images)

                                      images = imutils.resize(images, width=500)
                                      gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
                                      f, e, = os.path.splitext(root+"/"+item)

                                      # Remove item into dir
                                      os.remove(root+"/"+item)

                                      rects = detector(gray, 1)

                                      for (i, rect) in enumerate(rects):

                                          line = "\""+item+"~" + str(i)+ "\";"

                                          # determine the facial landmarks for the face region, then
                                          # convert the facial landmark (x, y)-coordinates to a NumPy
                                          # array
                                          shape = predictor(gray, rect)
                                          shape = face_utils.shape_to_np(shape)

                                          for (x, y) in shape:

                                              line = line + ";\""+str(x)+"~"+str(y)+"\""

                                          file.write(line.encode())
                                          file.write('\n'.encode()) 
                                  else:
                                        continue                     
               
for path in paths:
    resize( path )
