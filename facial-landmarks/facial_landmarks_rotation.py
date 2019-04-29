from PIL import Image
from imutils import face_utils
from numpy import array
from glob import glob

import os, sys
import argparse
import imutils
import dlib
import cv2
import json
import collections

import re
import io

import math 
import numpy as np

#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='a crop utility')

# Argument are :-- shape_predictor_68_face_landmarks.dat
parser.add_argument('-p', '--shape-predictor', type=str, nargs='?',
    help='path to facial landmark predictor')

parser.add_argument('--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')

parser.add_argument('-o', '--out_to_csv_file',type=str, nargs='?',help='if provided output will be writtent to csv(semicolon separated) otherwise to stdout. ')

FLAGS = parser.parse_args()



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FLAGS.shape_predictor)

print(FLAGS)

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/" ]

def resize( path ):
    items = os.listdir( path )

    if FLAGS.out_to_csv_file:

       with open(FLAGS.out_to_csv_file , 'wb' ) as file:

            for item in items:

                print(item)

                if item == '.DS_Store':
                    continue

                if os.path.isfile(path+item):
                
                    line = "\""+item+"\";"

                    # load the input image, resize it, and convert it to grayscale
                    images = cv2.imread(path+item)
                    images = imutils.resize(images, width=500)
                    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
                    f, e = os.path.splitext(path+item)
                    rects = detector(gray, 1)

                    xs = []
                    ys = []
                    for (i, rect) in enumerate(rects):
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        for (x, y) in shape:
                                
                            #line = line + ";\""+str(x)+"-"+str(y)+"\""
                            xs.append( x )
                            ys.append( y )

                        #file.write(line.encode())
                        #file.write('\n'.encode())

                        #detect angle of face around y axes
                        #transform it to to cx, cy
                        face_angle_around_y, deltaX, deltaY = angleOf(shape[31], p1y, p2x, p2y)

                        #determine the angle movement required  

                        #do the needed rotation if applicable
                        #double[] x = {0,  0,  0,  0,  0, 10, 20, 10};  //Create some test data
                        #double[] y = {0, 10, 20, 30, 40, 40, 30, 20};  //Create some test data
                        cx = 20 #//X-coord of center of rotation
                        cy = 30 #//Y-coord of center of rotation
                        angle = -45 * math.pi/180    #//convert 45 degrees to radians

                        #//Rotate the points
                        xs, ys = face_alignment( xs, ys, cx, cy, angle );

                        #save the result in csv

                        #//Display the results
                        ##out to display
                        ## convert dlib's rectangle to a OpenCV-style bounding box
                        ## [i.e., (x, y, w, h)], then draw the face bounding box
                        #(x, y, w, h) = face_utils.rect_to_bb(rect)
                        #cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # show the face number
                        #cv2.putText(images, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # loop over the (x, y)-coordinates for the facial landmarks
                        # and draw them on the image
                        blank_image = np.zeros((1024,1024,3), np.uint8)

                        for (x, y) in shape:
                            cv2.circle(blank_image, (250+x, 250+y), 1, (0, 0, 255), -1)

                        for i in range(0, len(xs)):
                            cv2.circle(blank_image, (250+int(xs[i]), 250+int(ys[i])), 1, (255, 255, 0), -1)


                        cv2.imshow("Output", blank_image)
                        cv2.waitKey(0)
                        #cv2.imwrite( os.path.join(FLAGS.out_to_dir, item+"_out.png"), blank_image )

#taken from https://stackoverflow.com/questions/2676719/calculating-the-angle-between-the-line-defined-by-two-points
def angleOf(p1x, p1y, p2x, p2y): 
    #// NOTE: Remember that most math has the Y axis as positive above the X.
    #// However, for screens we have Y as positive below. For this reason, 
    #// the Y values are inverted to get the expected results.
    deltaY = p1.y - p2.y
    deltaX = p2.x - p1.x
    result = math.toDegrees( math.atan2(deltaY, deltaX) ) 
    return (360d + result) if result < 0 else result, deltaX, deltaY

def face_alignment(x,y,cx,cy,angle):

    cos = math.cos(angle)
    sin = math.sin(angle)

    for n in range(0, len(x)):
        temp = ((x[n]-cx)*cos - (y[n]-cy)*sin) + cx
        y[n] = ((x[n]-cx)*sin + (y[n]-cy)*cos) + cy
        x[n] = temp

    return x, y


for path in paths:
    resize( path )
            # cv2.imshow("Output", images)
            # cv2.waitKey(0)
