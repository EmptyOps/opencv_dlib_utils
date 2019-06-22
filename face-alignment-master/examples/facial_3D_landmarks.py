import face_alignment
import numpy as np
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os, sys
import dlib
import imutils
import cv2


# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu',flip_input=False)

ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

# Instantiate the parser
parser = argparse.ArgumentParser(description='a crop utility')

ext = (".avi", ".mp4")

parser.add_argument('-d', '--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')
parser.add_argument('-o', '--out_to_csv_file',type=str, nargs='?',
    help='if provided output will be writtent to csv(semicolon separated) otherwise to stdout. ')
parser.add_argument('-ik', '--is_keep_extracted_image', action='store_true', help='A boolean True False')

detector = dlib.get_frontal_face_detector()

# input = io.imread('../test/__assets/img1.jpeg')
# preds = fa.get_landmarks(input)[-1]

FLAGS = parser.parse_args()

if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/" ]

def resize( path ):
    items = os.listdir( path )

    for filename in items:

        if (filename.endswith(ext)): #or .avi, .mpeg, whatever. 

            #single image only for testing "-vframes 50"

            NewDir = "../__data/__images/"+filename
            os.mkdir(NewDir)

            os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path+filename, filename ), os.path.join(NewDir, filename+"%d.jpeg")))     

            items1 = os.listdir(NewDir+"/")
            items1 = sorted(items1,key=lambda x: toint(os.path.splitext(x)[0]))

            with os.system(os.path.join( FLAGS.out_to_csv_file+"/__out", open(filename+"_dir.csv", 'wb' ))) as file:
            # with open(os.path.join( FLAGS.out_to_csv_file+"/__out_1", filename+"_dir.csv"), 'wb' ) as file:

                for item in items1:

                    if item == '.DS_Store':
                       continue

                    if (item.endswith('.jpeg')):
                      
                        # load the input image, resize it, and convert it to grayscale
                        images = cv2.imread(NewDir+"/"+item)
                        # images = cv2.imread( os.path.join( path, "images", item ) ) 
                        
                        try:
                          preds = fa.get_landmarks(images)[-1]
                        except:
                          # print("No faces were detected...!")
                          continue
                                      
                        #TODO: Make this nice
                        fig = plt.figure(figsize=plt.figaspect(.5))
                        ax = fig.add_subplot(1, 2, 1)
                        # ax.imshow(input)
                        ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                        ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
                        ax.axis('off')

                        ax = fig.add_subplot(1, 2, 2, projection='3d')
                        surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
                        ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
                        ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1], preds[17:22,2],color='blue')
                        ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
                        ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
                        ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
                        ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
                        ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,1], color='blue')
                        ax.plot3D(preds[48:,0]*1.2,preds[48:,1], preds[48:,2],color='blue' )

                        ax.view_init(elev=90., azim=90.)
                        ax.set_xlim(ax.get_xlim()[::-1])

                        # Remove item into dir
                        os.remove(root+"/"+item)

                        line = "\""+item+"~" + str(0)+ "\";"

                        for i in range(0, len(preds)):

                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y, z)-coordinates to a NumPy
                            # array

                            line = line + ";\""+str(preds[i][0])+"~"+str(preds[i][1])+"~"+str(preds[i][2])+"\""
                        file.write(line.encode())
                        file.write('\n'.encode()) 
        else:

              for root, dirs, files in os.walk(path+filename, topdown=False):

                  for name in files:
                     
                      if (name.endswith(ext)): #or .avi, .mpeg, whatever.

                         NewDir = "../__data/__images/"+name
                         if not os.path.exists(NewDir):
                            os.mkdir(NewDir)

                         #os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path+filename, name ), os.path.join(NewDir, name+"%d.jpeg")))     
                         os.system("ffmpeg -i {0} -f image2 -vf fps=fps=10 {1}".format( os.path.join( path+filename, name ), os.path.join(NewDir, name+"%d.jpeg")))     

                         items1 = os.listdir(NewDir+"/")

                         #hiren changed this on 19-06-2019
                         #with open(os.path.join( FLAGS.out_to_csv_file+"/__out", name+"_dir.csv"), 'wb' ) as file:
                         with open(os.path.join( FLAGS.out_to_csv_file+"/", name+"_dir.csv"), 'wb' ) as file:

                              for item in items1:

                                  if item == '.DS_Store':
                                     continue

                                  if (item.endswith('.jpeg')):

                                      # load the input image, resize it, and convert it to grayscale
                                      images = cv2.imread(NewDir+"/"+item)

                                      # f, e, = os.path.splitext(root+"/"+item)

                                      try:
                                        # print( "img " + item )
                                        preds = fa.get_landmarks(images);
                                        # print( len(preds) );
                                        # print(preds)

                                        for ( i, Prd ) in enumerate( preds ):

                                          #TODO: Make this nice
                                          fig = plt.figure(figsize=plt.figaspect(.5))
                                          ax = fig.add_subplot(1, 2, 1)
                                          # ax.imshow(input)
                                          ax.plot(Prd[0:17,0],Prd[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[17:22,0],Prd[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[22:27,0],Prd[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[27:31,0],Prd[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[31:36,0],Prd[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[36:42,0],Prd[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[42:48,0],Prd[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[48:60,0],Prd[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
                                          ax.plot(Prd[60:68,0],Prd[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
                                          ax.axis('off')

                                          ax = fig.add_subplot(1, 2, 2, projection='3d')
                                          surf = ax.scatter(Prd[:,0]*1.2,Prd[:,1],Prd[:,2],c="cyan", alpha=1.0, edgecolor='b')
                                          ax.plot3D(Prd[:17,0]*1.2,Prd[:17,1], Prd[:17,2], color='blue' )
                                          ax.plot3D(Prd[17:22,0]*1.2,Prd[17:22,1],Prd[17:22,2],color='blue')
                                          ax.plot3D(Prd[22:27,0]*1.2,Prd[22:27,1],Prd[22:27,2], color='blue')
                                          ax.plot3D(Prd[27:31,0]*1.2,Prd[27:31,1],Prd[27:31,2], color='blue')
                                          ax.plot3D(Prd[31:36,0]*1.2,Prd[31:36,1],Prd[31:36,2], color='blue')
                                          ax.plot3D(Prd[36:42,0]*1.2,Prd[36:42,1],Prd[36:42,2], color='blue')
                                          ax.plot3D(Prd[42:48,0]*1.2,Prd[42:48,1],Prd[42:48,1], color='blue')
                                          ax.plot3D(Prd[48:,0]*1.2,Prd[48:,1], Prd[48:,2],color='blue' )

                                          ax.view_init(elev=90., azim=90.)
                                          ax.set_xlim(ax.get_xlim()[::-1])

                                          # Remove item into dir
                                          if not FLAGS.is_keep_extracted_image:
                                            os.remove(NewDir+"/"+item)

                                          # for (i,Pre) in enumerate(Prd):

                                          line = "\""+item+"~" + str(i)+ "\";"

                                          for i in range(0, len(Prd)):

                                              # determine the facial landmarks for the face region, then
                                              # convert the facial landmark (x, y, z)-coordinates to a NumPy
                                              # array

                                              line = line + ";\""+str(Prd[i][0])+"~"+str(Prd[i][1])+"~"+str(Prd[i][2])+"\""
                                          file.write(line.encode())
                                          file.write('\n'.encode()) 

                                      except:
                                        continue
                                  else:
                                        continue   

               
for path in paths:
    resize( path )
