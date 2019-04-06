from PIL import Image
from imutils import face_utils
import os, sys
import argparse
import imutils
import dlib
import cv2

import re
import io

from glob import glob

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
parser.add_argument('--out_to_csv_file',type=str, nargs='?',help='if provided output will be writtent to csv(semicolon separated) otherwise to stdout. ')
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
    for item in items:
        print(item)
        if item == '.DS_Store':
            continue

        if os.path.isfile(path+item):
        
            # load the input image, resize it, and convert it to grayscale
            images = cv2.imread(path+item)
            images = imutils.resize(images, width=500)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
            f, e = os.path.splitext(path+item)
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                #out to display
                """
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the face number
                cv2.putText(images, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(images, (x, y), 1, (0, 0, 255), -1)

                cv2.imshow("Output", images)
                cv2.waitKey(0)
                """

                #out_to_csv_file
                if FLAGS.out_to_csv_file:

                    dirs_orig = glob( FLAGS.dir_to_process + "/*/" )

                    with open( FLAGS.out_to_csv_file, 'wb' ) as file:
                    
                        for dir in path:
                            actual_dir = os.path.basename( os.path.dirname(dir) )

                            if "text" in res:
                                if actual_dir in res["text"]:
                                
                                    out_dir = os.path.join( FLAGS.output_dir, actual_dir ) if len(dirs_orig) > 0 else FLAGS.output_dir
                                    items = os.listdir( out_dir )
                                    for item in items:

                                        #drop bad characters 
                                        with io.open( os.path.join( out_dir, item ),'r',encoding='utf-8',errors='ignore') as infile, \
                                             io.open( os.path.join( out_dir, item ) + 'd_parsed.txt','w',encoding='ascii',errors='ignore') as outfile:
                                                for line in infile:
                                                    print(*line.split(), file=outfile)

                                        #write to csv
                                        with open( os.path.join( out_dir, item ) + 'd_parsed.txt', 'r', errors='ignore' ) as myfile:
                                            strv = "" + myfile.read() + "".strip() 
                                            print( strv )
                                        
                                            if FLAGS.extract_text_regex_char_whitelist:
                                                strv = re.sub("[^"+FLAGS.extract_text_regex_char_whitelist+"]", "", strv)
                                                
                                            if FLAGS.extract_text_replace_newline_with_space:
                                                strv = strv.replace('\n', ' ')
                                                
                                            print( strv )
                                        
                                            fname = os.path.splitext( item )[0]
                                            line = "\""+actual_dir+"\";\""+fname+"\";\"'"+ strv +"\"" 
                                            file.write(line.encode())
                                            file.write('\n'.encode())
                            


for path in paths:
    resize( path )
            # cv2.imshow("Output", images)
            # cv2.waitKey(0)