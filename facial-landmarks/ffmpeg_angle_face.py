from imutils import face_utils

import os, sys
import argparse
import imutils
import dlib
import cv2
from PIL import Image
import shutil

from numpy import array
import numpy as np


ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

ext = (".avi", ".mp4", ".mpeg")

FLAGS = None
detector = None
predictor = None

def toint(str):
  try:
    return int(str)
  except Exception as e:
    return 0
  else:
    pass
  finally:
    pass

def num_float_first(s):
    try:
        return float(s)
    except ValueError:
        return int(s)

def to_x_y_0(strng):
    cords = strng.split('~')

    return [ num_float_first(cords[0]), num_float_first(cords[1]) ]

def to_x_y(strng):
    # cords = strng.split('~')

    # return [ num_float_first(cords[0]), num_float_first(cords[1]) ]
    return [ strng[0], strng[1] ]

def get_rotation(csv_row):

      rotation_vector = []

      #
      lStart = to_x_y( csv_row[17] )
      lEnd = to_x_y( csv_row[21] )
      rStart = to_x_y( csv_row[22] )
      rEnd = to_x_y( csv_row[26] )

      leftEyePts = np.array( [lStart,lEnd] )
      rightEyePts = np.array( [rStart,rEnd] )

      # compute the center of mass for each eye
      leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
      rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

      # compute the angle between the eye centroids
      dY = rightEyeCenter[1] - leftEyeCenter[1]
      dX = rightEyeCenter[0] - leftEyeCenter[0]
      rotation_vector.append( np.degrees(np.arctan2(dY, dX)) ) # - 180 )

      #
      l = to_x_y( csv_row[1] )
      n = to_x_y( csv_row[33] )
      r = to_x_y( csv_row[15] )

      # compute the distances
      dAngle = (r[0]-n[0]) - (n[0]-l[0])
      dRatio = (r[0]-n[0]) * 2 if (r[0]-n[0]) > (n[0]-l[0]) else (n[0]-l[0]) * 2
      rotation_vector.append( np.degrees(np.arctan2(dAngle, dRatio)) ) # - 180 )

      rotation_vector.append( 0 )

      return rotation_vector

def face_coords(csv_row, dirangle, width, height):

    face_cords = []

    #face coords 
    xtl, _ = to_x_y_0( csv_row[2] )
    _, ytl = to_x_y_0( csv_row[21] )
    _, ybr = to_x_y_0( csv_row[10] )
    xbr, _ = to_x_y_0( csv_row[18] )   
    _, ytr = to_x_y_0( csv_row[26] )

    if ytl < ytr:
        ytr = ytl
    else:
        ytl = ytr

    face_cords.append( xtl )
    face_cords.append( ytl )
    face_cords.append( xbr )
    face_cords.append( ybr )        

    #
    magnitude = face_cords[2] - face_cords[0] if (face_cords[2] - face_cords[0]) > (face_cords[3]-face_cords[1]) else face_cords[3]-face_cords[1]    
    if magnitude <= 0:
        face_cords = [ 0, 0, 0, 0 ]
        return face_cords

    face_image_padding = int( 20 * (magnitude/25) ) 

    #
    if dirangle < -2:
        face_image_padding = face_image_padding + int( 10 * (magnitude/25) )
    elif dirangle > 2:
        face_image_padding = face_image_padding + int( 10 * (magnitude/25) )    

    #
    # if dirangle < 0:
    xbr = xbr + face_image_padding
    # elif dirangle > 0:
    xtl = xtl - face_image_padding if xtl - face_image_padding > 0 else 0

    ytl = ytl - face_image_padding
    ybr = ybr + face_image_padding    

    xtr = xbr
    xbl = xtl 
    ybl = ybr

    #
    n = to_x_y_0( csv_row[35] )
    if xbr <= n[0]:
        xbr = n[0] + ((n[0]-xtl)/5)

    if ybr <= n[1]:
        ybr = n[0] + ((n[0]-xtl)/5)

    #
    xtl = xtl if xtl > 0 else 0
    ytl = ytl if ytl > 0 else 0
    xbr = xbr if xbr < width else width     # im.size[0] else im.size[0]
    ybr = ybr if ybr < height else height   # im.size[1] else im.size[1]

    #hiren added below line on 08-03-2020 as it seems like a bug as forgot to reset face_cords so just reset it
    face_cords = []

    face_cords.append( xtl )
    face_cords.append( ytl )
    face_cords.append( xbr )
    face_cords.append( ybr )        

    return face_cords

def face_wh(csv_row, dirangle, width, height):

    face_cords = face_coords(csv_row, dirangle, width, height)

    return face_cords[2]-face_cords[0], face_cords[3]-face_cords[1]

def scale_if_small(imgpath, min_width_limit, savepath, scale_by=2.5):
  im = Image.open( imgpath )
  if im.size[0] < min_width_limit:
    basewidth = im.size[0] * scale
    wpercent = (basewidth/float(im.size[0]))
    hsize = int((float(im.size[1])*float(wpercent)))
    im = im.resize((basewidth,hsize), Image.ANTIALIAS)
    im.save(savepath) 

def resize( path, FLAGSLcl, detector, predictor ):

    #
    FLAGS = FLAGSLcl

    if not os.path.exists( FLAGS.output_dir_for_csv_files ):
      os.mkdir( FLAGS.output_dir_for_csv_files )

    if not os.path.exists( FLAGS.extracted_images_dir ):
      os.mkdir( FLAGS.extracted_images_dir )

    is_passed = False
    is_single_file = False

    #check if video file path then treat it as such 
    if os.path.isfile( path ):
      is_single_file = True
      items = [ os.path.basename(path) ]
      path = os.path.dirname(path)
    else:
      items = os.listdir(path)

    pathOrg = path
    for filenameGlb in items:

        if True or (filename.endswith(ext)): #or .avi, .mpeg, whatever. 

          files = []
          prefix = ""
          if (filenameGlb.endswith(ext)): #or .avi, .mpeg, whatever. 
            files.append( filenameGlb )
          elif os.path.isdir( os.path.join( pathOrg, filenameGlb ) ):
            prefix = filenameGlb+"_"
            path = os.path.join( pathOrg, filenameGlb )
            files = os.listdir(path)
          elif FLAGS.is_skip_dirs_and_other_files:
            continue
          else:
            raise Exception("Unsupported filetype '"+filenameGlb+"' found")

          for filename in files:

            #assert duplicates 
            if os.path.exists( os.path.join( FLAGS.output_dir_for_csv_files, prefix + filename+".csv") ):
              if FLAGS.is_skip_duplicates:
                continue
              elif not FLAGS.is_overwrite_duplicates:
                raise Exception("Fatal error duplicate file "+prefix + filename+" detected. Terminating script")

            # outdir = os.path.join( path, filename+"_dir" )
            # os.makedirs( outdir )
            # os.system( "ffmpeg -i {0} -f image2 -vf fps=fps=1 {1}".format( os.path.join( path, filename ), os.path.join( path, "output%d.jpeg" ) ) )
            imgdir = os.path.join(FLAGS.extracted_images_dir, prefix + filename)
            if not os.path.exists( imgdir ):
              os.mkdir( imgdir )

            #single image only for testing "-vframes 50"
            os.system( "ffmpeg -i {0} -f image2 -vf fps=fps={1} {2}".format( os.path.join( path, filename ), FLAGS.fps, os.path.join( imgdir, "%d.jpeg" ) ) )     

            print("sorting by names")
            items1 = os.listdir( imgdir ) 
            items1 = sorted(items1,key=lambda x: toint(os.path.splitext(x)[0]))
            print("sorting by names done")

            # with os.system(os.path.join( outdir, open(filename+"_dir.csv", 'wb' ))) as file:
            with open(os.path.join( FLAGS.output_dir_for_csv_files, prefix + filename+".csv"), 'wb' ) as file:

                itemcnt = 0
                totitem = len(items1)

                for item in items1:

                    itemcnt += 1
                    if not is_passed and FLAGS.skip_if_filter_fails_initially > 0 and (itemcnt/totitem) * 100 >= FLAGS.skip_if_filter_fails_initially:
                      print("returning after no pass even after ", FLAGS.skip_if_filter_fails_initially, " percent checks")
                      shutil.rmtree(imgdir)
                      return False

                    if item == '.DS_Store':
                       continue

                    if (item.endswith('.jpeg')):

                        #
                        imgpath = os.path.join( imgdir, item )
                        scale_if_small(imgpath, 33, imgpath)

                        # load the input image, resize it, and convert it to grayscale
                        images = cv2.imread( imgpath ) 
                        height, width, channels = images.shape

                        # images = imutils.resize(images, width=500)
                        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
                        f, e = os.path.splitext( imgpath )

                        # Remove item into dir
                        if not FLAGS.is_keep_extracted_image:
                          os.remove( imgpath )

                        rects = detector(gray, 1)

                        if len(rects) <= 0:
                          os.remove( imgpath )
                          print("No face detected. Skipping...")

                        for (i, rect) in enumerate(rects):

                            line = "\""+item+"~"+str(i)+"\";"
                            if FLAGS.skip_if_filter_fails_initially > 0:
                              csv_row_tmp = []
                              csv_row_tmp.append( ""+item+"~"+str(i)+"" )
                              csv_row_tmp.append( "" )

                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            #
                            rot_m = get_rotation(shape)
                            angle = rot_m[0]     
                            t = round( rot_m[1] / 10 )
                            dirangle = abs( t ) if t == -0 else t    
                            if dirangle < -1 or dirangle > 1:
                                print("dirangle " + str(dirangle) + " out of limit. Skipping...")
                                if os.path.exists(imgpath):
                                  os.remove( imgpath )
                                continue

                            print( "rotate angle " + str(angle) + " dirangle " + str(dirangle) + " item " + item )


                            for (x, y) in shape:

                                line = line + ";\""+str(x)+"~"+str(y)+"\""

                                if FLAGS.skip_if_filter_fails_initially > 0:
                                  csv_row_tmp.append( ""+str(x)+"~"+str(y)+"" )

                            if FLAGS.skip_if_filter_fails_initially > 0:
                              w, h = face_wh( csv_row_tmp, dirangle, width, height )

                              if w < FLAGS.filter_int_val_1 or h < FLAGS.filter_int_val_2:
                                continue
                              else:
                                is_passed = True

                            file.write(line.encode())
                            file.write('\n'.encode()) 
        
            # Remove item into dir
            if not FLAGS.is_keep_video_file:
              os.remove(os.path.join( path, filename ) )
        else:

              if FLAGS.is_skip_dirs_and_other_files:
                continue
          
              for root, dirs, files in os.walk(path+filename, topdown=False):

                  for name in files:
                     
                      if (name.endswith(ext)): #or .avi, .mpeg, whatever.

                         os.system("ffmpeg -i {0} -f image2 -vf fps=fps="+FLAGS.fps+" {1}".format( os.path.join( path+filename, name ), os.path.join(path+filename, name+"%d.jpeg" )))     

                         items1 = os.listdir(root+"/")

                         with open(os.path.join( FLAGS.output_dir_for_csv_files, name+"_dir.csv"), 'wb' ) as file:

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
               
    if is_single_file:
        if is_passed:
          return True
        else:
          return False


if __name__ == '__main__':
  import argparse

  # Instantiate the parser
  parser = argparse.ArgumentParser(description='a crop utility')

  # Argument are :-- shape_predictor_68_face_landmarks.dat
  parser.add_argument('-p', '--shape-predictor', type=str, nargs='?',
      help='path to facial landmark predictor')

  parser.add_argument('-d', '--dir_to_process', type=str, nargs='?',
                      help='dir_to_process')
  parser.add_argument('-ik', '--is_keep_extracted_image', action='store_true', help='A boolean True False')

  parser.add_argument('-f', '--fps', type=str, nargs='?',
                      help='image extract per fps')
  parser.add_argument('-ikv', '--is_keep_video_file', action='store_true', help='A boolean True False')
  parser.add_argument('-isd', '--is_skip_dirs_and_other_files', action='store_true', help='A boolean True False')
  parser.add_argument('-isu', '--is_skip_duplicates', action='store_true', help='A boolean True False')
  parser.add_argument('-iod', '--is_overwrite_duplicates', action='store_true', help='A boolean True False')
  

  parser.add_argument('--filter_int_val_1',type=int, nargs='?',help='filter_int_val_1')
  parser.add_argument('--filter_int_val_2',type=int, nargs='?',help='filter_int_val_2')

  parser.add_argument('-oc', '--output_dir_for_csv_files', type=str, nargs='?',
                      help='output_dir_for_csv_files')
  parser.add_argument('-oi', '--extracted_images_dir', type=str, nargs='?',
                      help='extracted_images_dir')

  parser.add_argument('-skp', '--skip_if_filter_fails_initially', type=int, nargs='?', help=' pass -1 to ignore')


  FLAGS = parser.parse_args()

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(FLAGS.shape_predictor)

  if FLAGS.dir_to_process == "":
      paths = []  #specify static here
  else:
      paths = [FLAGS.dir_to_process+"/"]

  for path in paths:
      resize( path, FLAGS, detector, predictor )
