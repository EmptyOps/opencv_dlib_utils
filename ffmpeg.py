import os, sys
import numpy as np
import argparse
import dlib
import subprocess

#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='a crop utility')

parser.add_argument('--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')

FLAGS = parser.parse_args()

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/" ]

def resize( path ):
    items = os.listdir( path )

    for filename in items:

        if (filename.endswith('.mp4')): #or .avi, .mpeg, whatever. 

            outdir = os.path.join( path, filename+"_dir" )
            os.makedirs( outdir )
            os.system( "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path, filename ), os.path.join( outdir, "output%d.jpeg" ) ) )
             
        else:
            continue

for path in paths:
    resize( path )