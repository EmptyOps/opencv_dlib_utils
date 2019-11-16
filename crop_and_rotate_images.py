from PIL import Image
from imutils import face_utils
from numpy import array
from glob import glob

import os, sys
import argparse
import imutils
import cv2
import json
import collections
import dlib

import re
import io
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

import csv

from copy import copy

from pyagender import PyAgender

agender = PyAgender() 

#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='a utility')

parser.add_argument('-d', '--dir_to_process', type=str, nargs='?',
                    help='dir_to_process')
parser.add_argument('-i', '--imgages_dir', type=str, nargs='?',
                    help='imgages_dir')
parser.add_argument('-x', '--img_extension', type=str, nargs='?',
                    help='image extension')
parser.add_argument('-v', '--verification_val', type=str, nargs='?',
                    help='verification value')
parser.add_argument('-n', '--gender_man_lbl', type=str, nargs='?',
                    help='gender man label value')


parser.add_argument('-o', '--out_to_dir',type=str, nargs='?',help='if provided output will be written to csv(semicolon separated) otherwise to stdout. ')
parser.add_argument('-s', '--is_save_image', action='store_true', help='A boolean True False')

# Argument are :-- shape_predictor_68_face_landmarks.dat
parser.add_argument('-p', '--shape-predictor', type=str, nargs='?', help='path to facial landmark predictor')

parser.add_argument('-m', '--merge_and_fit', action='store_true', help='A boolean True False')

FLAGS = parser.parse_args()

print(FLAGS)

"""
This file needs refactoring and especially removing the unused functions 
"""


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FLAGS.shape_predictor)


if FLAGS.dir_to_process == "":
    paths = []  #specify static here
else:
    paths = [FLAGS.dir_to_process+"/" ]



def num_float_first(s):
    try:
        return float(s)
    except ValueError:
        return int(s)

def to_x_y(strng):
    cords = strng.split('~')

    # print(strng)
    # print(cords)
    #return np.array( [ float(cords[0]), float(cords[1]) ] )
    return [ num_float_first(cords[0]), num_float_first(cords[1]) ]

def to_x_y_z(strng):
    cords = strng.split('~')

    # print(strng)
    # print(cords)
    #return np.array( [ float(cords[0]), float(cords[1]) ] )
    return [ num_float_first(cords[0]), num_float_first(cords[1]), num_float_first(cords[2]) ]

def get_rotation_and_translation_matrix(csv_row, scale, imgpath_org):

    is_arc_rotation = True
    
    # Read Image
    imgpath = imgpath_org  

    if is_arc_rotation:
        rotation_vector = []

        #
        # (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        # (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        lStart = to_x_y( csv_row[19] )
        lEnd = to_x_y( csv_row[23] )
        rStart = to_x_y( csv_row[24] )
        rEnd = to_x_y( csv_row[28] )

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
        l = to_x_y( csv_row[3] )
        n = to_x_y( csv_row[35] )
        r = to_x_y( csv_row[17] )
 
        # compute the distances
        dAngle = (r[0]-n[0]) - (n[0]-l[0])
        dRatio = (r[0]-n[0]) * 2 if (r[0]-n[0]) > (n[0]-l[0]) else (n[0]-l[0]) * 2
        rotation_vector.append( np.degrees(np.arctan2(dAngle, dRatio)) ) # - 180 )

        rotation_vector.append( 0 )

        return rotation_vector, [], imgpath, "", "1_" + os.path.basename(imgpath), os.path.basename(os.path.dirname(imgpath)) + ".1/"

def scale_if_small(imgpath, min_width_limit, savepath, scale_by=2.5):
  im = Image.open( imgpath )
  if im.size[0] < min_width_limit:
    basewidth = im.size[0] * scale
    wpercent = (basewidth/float(im.size[0]))
    hsize = int((float(im.size[1])*float(wpercent)))
    im = im.resize((basewidth,hsize), Image.ANTIALIAS)
    im.save(savepath) 

def do_rotation(csv_row, is_man):

    #
    # print('to_x_y(csv_row[2])')
    # print(to_x_y(csv_row[2]))
    # print(csv_row[2])

    #
    scale = 1   #3
    is_crop_main_expressions_only = True
    is_detect_shape_after_rotate = True
    is_visualize_detected_face = False
    is_fit_parts_to_size = True
    is_fit_parts_to_size_with_customization = False


    # load the input image, resize it, and convert it to grayscale
    imgpath_org = FLAGS.imgages_dir + csv_row[0] + "." + FLAGS.img_extension
    if not os.path.exists(imgpath_org):
        print(imgpath_org)
        got_here
        return

    #
    scale_if_small(imgpath_org, 33, imgpath_org)

    images = cv2.imread( imgpath_org )

    # images = imutils.resize(images, width=500)
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
    f, e = os.path.splitext( imgpath_org )
    rects = detector(gray, 1)

    #skip if no face is detected
    if len(rects) <= 0:
        print( "Nooooooooooooooo Faceeeeeeeeeeeeeeeeee found in Imageeeeeeeeeeeeeeeeee. Skipping..." )
        return

    #if more than one face detected in cropped image then simply continue for now 
    if len(rects) >= 2:
        print( "more than one face detected, skipping..." )
        return

    #
    new_csv_row = []
    new_csv_row.append( csv_row[0] )
    new_csv_row.append( csv_row[2] )
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            new_csv_row.append( ""+str(x)+"~"+str(y)+"~0" )

    csv_row = array( new_csv_row )
    ##########################################################################################################


    #
    org_csv_row = []
    org_csv_row.append( csv_row[0] )
    org_csv_row.append( csv_row[1] )
    pLeft = [to_x_y(csv_row[2])]
    for i in range(3,69):
        org_csv_row.append( csv_row[i] )
        pLeft.append( to_x_y(csv_row[i]) )

    pLeft = np.array( pLeft )

    #
    is_do_custom_rotation = False

    #
    rot_m, tran_m, imgpath, _, imgname, imgdirname = get_rotation_and_translation_matrix(csv_row, scale, imgpath_org)
    if imgname == None:
        return

    imgname = imgname.replace(FLAGS.img_extension, "png")

    if FLAGS.merge_and_fit and ( not os.path.exists( FLAGS.imgages_dir + "/" + imgdirname + "_parts/eye_" + imgname ) or not os.path.exists( FLAGS.imgages_dir + "/" + imgdirname + "_parts/mouth_" + imgname ) ): 
        print("MERGE AND FITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT, Image not found. Skipping...")
        return

    #
    angle = rot_m[0]    #round( ( (rot_m[0][0]/math.pi) * 180 ) )
    t = round( rot_m[1] / 10 )
    dirangle = abs( t ) if t == -0 else t    #str( round( ( (rot_m[1][0]/math.pi) * 180 ) / 10 ) )

    if dirangle < -1 or dirangle > 1:
        print("dirangle " + str(dirangle) + " out of limit. Skipping...")
        return

    print( "rotate angle " + str(angle) + " dirangle " + str(dirangle) )

    im = Image.open(imgpath)

    #crop face part
    im_expression_eye = None
    im_expression_mouth = None
    eye_cords = []
    mouth_cords = []
    face_cords = []

    #face coords 
    xtl, _, _ = to_x_y_z( csv_row[2] )
    _, ytl, _ = to_x_y_z( csv_row[21] )
    _, ybr, _ = to_x_y_z( csv_row[10] )
    xbr, _, _ = to_x_y_z( csv_row[18] )   #to_x_y_z( csv_row[28] )   #( csv_row[37] )
    _, ytr, _ = to_x_y_z( csv_row[26] )

    if ytl < ytr:
        ytr = ytl
    else:
        ytl = ytr

    face_cords.append( xtl )
    face_cords.append( ytl )
    face_cords.append( xbr )
    face_cords.append( ybr )        

    #
    magnitude = face_cords[2] - face_cords[0] if (face_cords[2] - face_cords[0]) > (face_cords[3]-face_cords[1]) else face_cords[3]-face_cords[1]    #should actually calculate it later by calculating distance between most wide points out of four 
    if magnitude <= 0:
        #raise Exception( "unexpected_magnitude_found_intended_to_be_crashed_here" )
        print( "SKIPPING DUE TO ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR..." )
        print( "unexpected_magnitude_found_intended_to_be_crashed_here" )
        return

    face_image_padding = int( 20 * (magnitude/25) ) #20

    #
    if dirangle < -2:
        face_image_padding = face_image_padding + int( 10 * (magnitude/25) )
    elif dirangle > 2:
        face_image_padding = face_image_padding + int( 10 * (magnitude/25) )    #xtl = xtl - face_image_padding if xtl - face_image_padding > 0 else 0

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
    n = to_x_y( csv_row[35] )
    if xbr <= n[0]:
        xbr = n[0] + ((n[0]-xtl)/5)

    if ybr <= n[1]:
        ybr = n[0] + ((n[0]-xtl)/5)

    #
    xtl = xtl if xtl > 0 else 0
    ytl = ytl if ytl > 0 else 0
    xbr = xbr if xbr < im.size[0] else im.size[0]
    ybr = ybr if ybr < im.size[1] else im.size[1]

    face_cords.append( xtl )
    face_cords.append( ytl )
    face_cords.append( xbr )
    face_cords.append( ybr )        

    #
    if int(xtl) == int(xbr) or int(ytl) == int(ybr):
        print( "zero size image detected, skipping..." )
        return

    #
    if not FLAGS.merge_and_fit and is_detect_shape_after_rotate == True: 
        im = im.crop( (int(xtl), int(ytl), xbr, ybr ) )
        im = im.rotate(angle)        

        #TODO temp 
        try: 
            # print( im.size )
            # print( int(xtl), int(ytl), xbr, ybr )
            im.save( FLAGS.out_to_dir + "/tmp/" + imgname )        

            # load the input image, resize it, and convert it to grayscale
            images = cv2.imread( FLAGS.out_to_dir + "/tmp/" + imgname )
            # images = imutils.resize(images, width=500)
            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
            f, e = os.path.splitext( FLAGS.out_to_dir + "/tmp/" + imgname )
            rects = detector(gray, 1)

            #skip if no face is detected
            if len(rects) <= 0:
                print( "Zerooooo00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 face detected in cropped box, there is something wrong, should be fixed at earliest. Skipping..." )
                return

            # #if more than one face detected in cropped image then simply continue for now 
            # if len(rects) >= 2:
            #     print( "more than one face detected, skipping..." )
            #     im.show()
            #     sdfhsfhdfkhj
            #     return

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                ci = 2
                for (x, y) in shape:
                    csv_row[ci] = ""+str(x)+"~"+str(y)+"~0"
                    ci = ci + 1

                #for now use the first face only
                break

            # im.show()
            # cv2.waitKey(0)
            # input("Press Enter to continue...")
        except Exception as e:
            print("SKIPPING DUE TO ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR...")
            print(e)
            xdfcjdkfdfhj
            return

    if is_crop_main_expressions_only == False:
        # zbl = zbr
        print( "(int(xtl), int(ytl), xbr, ybr ) " + str(int(xtl)) + " " + str(int(ytl)) + " " + str(xbr) + " " + str(ybr) + " " )
        print(n)
        im = im.crop( (int(xtl), int(ytl), xbr, ybr ) )
    else:
        if not FLAGS.merge_and_fit:
            if is_detect_shape_after_rotate == True: 
                #update face coords 
                face_cords.append( 0 )
                face_cords.append( 0 )
                face_cords.append( im.size[0] )
                face_cords.append( im.size[1] )        

            magnitude = face_cords[2] - face_cords[0] if (face_cords[2] - face_cords[0]) > (face_cords[3]-face_cords[1]) else face_cords[3]-face_cords[1]    #should actually calculate it later by calculating distance between most wide points out of four 
            if magnitude <= 0:
                #raise Exception( "unexpected_magnitude_found_intended_to_be_crashed_here" )
                print( "SKIPPING DUE TO ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR..." )
                print( "unexpected_magnitude_found_intended_to_be_crashed_here" )
                return

            padding = magnitude/25
            print( "magnitude " + str(magnitude) + " padding " + str(padding) )

            #eyes area 
            xtl, _, _ = to_x_y_z( csv_row[19] ) 
            _, ytl, _ = to_x_y_z( csv_row[21] )
            xbr, _, _ = to_x_y_z( csv_row[28] ) 
            _, ybr, _ = to_x_y_z( csv_row[48] ) 
            _, ytr, _ = to_x_y_z( csv_row[26] )

            xtl = xtl - padding # ((xbr-xtl)/5)
            xbr = xbr + padding # ((xbr-xtl)/5)
            ytl = ytl - padding # ((ybr-ytl)/5)
            ybr = ybr + padding * 3 #
            ytr = ytr - padding # ((ybr-ytl)/5)


            if ytl < ytr:
                ytr = ytl
            else:
                ytl = ytr

            # if dirangle < -2:
            #     xbr = xbr + 20
            # elif dirangle > 2:
            #     xtl = xtl - 20 if xtl - 20 > 0 else 0

            xtr = xbr
            xbl = xtl 
            ybl = ybr

            print( "eyes area 0 (int(xtl), int(ytl), xbr, ybr ) " + str(int(xtl)) + " " + str(int(ytl)) + " " + str(xbr) + " " + str(ybr) + " " )

            #
            n = to_x_y( csv_row[35] )
            if xbr <= n[0]:
                xbr = n[0] + ((n[0]-xtl)/5)

            # if ybr <= n[1]:
            #     ybr = n[0] + ((n[0]-xtl)/5)

            print( "eyes area (int(xtl), int(ytl), xbr, ybr ) " + str(int(xtl)) + " " + str(int(ytl)) + " " + str(xbr) + " " + str(ybr) + " " )
            print(n)
            im_expression_eye = im.crop( (int(xtl), int(ytl), xbr, ybr ) )
            if is_fit_parts_to_size == True:
                if not os.path.exists( FLAGS.imgages_dir + "/" + imgdirname ):
                    os.mkdir( FLAGS.imgages_dir + "/" + imgdirname )

                if not os.path.exists( FLAGS.imgages_dir + "/" + imgdirname + "_parts" ):
                    os.mkdir( FLAGS.imgages_dir + "/" + imgdirname + "_parts" )                

                im_expression_eye.save( FLAGS.imgages_dir + "/" + imgdirname + "_parts/eye_" + imgname ) 

            eye_cords.append( xtl )
            eye_cords.append( ytl )
            eye_cords.append( xbr )
            eye_cords.append( ybr )
            # im.show()
            # im_expression_eye.show()
            # input("Press Enter to continue...")
            # sdfsdjfhgdsjhfg

            #mouth area 
            xtl, _, _ = to_x_y_z( csv_row[50] )
            _, ytl, _ = to_x_y_z( csv_row[52] )
            xbr, _, _ = to_x_y_z( csv_row[56] ) 
            _, ytr, _ = to_x_y_z( csv_row[54] ) 
            _, ybr, _ = to_x_y_z( csv_row[59] ) 

            xtl = xtl - padding * 2 # ((xbr-xtl)/5)
            xbr = xbr + padding * 2 # ((xbr-xtl)/5)
            ytl = ytl - padding * 1 # ((ybr-ytl)/5)
            ybr = ybr + padding * 2 #
            ytr = ytr - padding * 1 # ((ybr-ytl)/5)

            if ytl < ytr:
                ytr = ytl
            else:
                ytl = ytr

            if dirangle < -2:
                xbr = xbr + 20
            elif dirangle > 2:
                xtl = xtl - 20 if xtl - 20 > 0 else 0

            xtr = xbr
            xbl = xtl 
            ybl = ybr

            #
            n = to_x_y( csv_row[35] )
            if xbr <= n[0]:
                xbr = n[0] + ((n[0]-xtl)/5)

            if ybr <= n[1]:
                ybr = n[0] + ((n[0]-xtl)/5)


            print( "mouth area (int(xtl), int(ytl), xbr, ybr ) " + str(int(xtl)) + " " + str(int(ytl)) + " " + str(xbr) + " " + str(ybr) + " " )
            
            #mouth is wider than eyes or if eyes is below 20 in width or mouth is below 7 in width then continue
            if (xbr-xtl) > (eye_cords[2]-eye_cords[0]) or (eye_cords[2]-eye_cords[0]) < 20 or (xbr-xtl) < 7:
                print( "SKIPPING DUE TO ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR..." )
                print( "mouth width is inappropriate" )
                return

            print(n)
            im_expression_mouth = im.crop( (int(xtl), int(ytl), xbr, ybr ) )
            if is_fit_parts_to_size == True:
                im_expression_mouth.save( FLAGS.imgages_dir + "/" + imgdirname + "_parts/mouth_" + imgname ) 
                return
            else:
                mouth_cords.append( xtl )
                mouth_cords.append( ytl )
                mouth_cords.append( xbr )
                mouth_cords.append( ybr )
                # im.show()
                # im_expression_mouth.show()
                # input("Press Enter to continue...")
                # sdfsdjfhgdsjhfg
        else:
            im_expression_eye = Image.open( FLAGS.imgages_dir + "/" + imgdirname + "_parts/eye_" + imgname )
            im_expression_mouth = Image.open( FLAGS.imgages_dir + "/" + imgdirname + "_parts/mouth_" + imgname )

    #rotate image
    if not angle == 0:
        if is_detect_shape_after_rotate == False: 
            if is_crop_main_expressions_only == False:
                im = im.rotate(angle)
            else:
                # im = im.rotate(angle)
                # im.show()
                # cv2.waitKey(0)
                im_expression_eye = im_expression_eye.rotate(angle)
                im_expression_mouth = im_expression_mouth.rotate(angle)

    #save 
    dirangle = str(dirangle) 

    if not os.path.exists( FLAGS.imgages_dir + "/tmp/" ):
        os.mkdir( FLAGS.imgages_dir + "/tmp/" )

    #check gender 
    # if os.path.exists( FLAGS.imgages_dir + "/tmp/" + imgname + "_g_f.txt" ):
    if not is_man:
        dirangle = dirangle + "_f"
    # elif os.path.exists( FLAGS.imgages_dir + "/tmp/" + imgname + "_g_m.txt" ):
    else:
        dirangle = dirangle + "_m"
    # else:
    #     faces = agender.detect_genders_ages( cv2.imread( FLAGS.out_to_dir + "/tmp/" + imgname ) )

    #     #TODO temp
    #     print( faces )
    #     im = Image.open(FLAGS.out_to_dir + "/tmp/" + imgname)
    #     im.show()
    #     input("Press Enter to continue...")

    #     if len(faces) > 0:
    #         if faces[0]["gender"] <= 0.50:
    #             with open(FLAGS.imgages_dir + "/tmp/" + imgname + "_g_m.txt", "w") as text_file:
    #                 text_file.write("1")

    #             dirangle = dirangle + "_m"
    #         else:
    #             with open(FLAGS.imgages_dir + "/tmp/" + imgname + "_g_f.txt", "w") as text_file:
    #                 text_file.write("1")

    #             dirangle = dirangle + "_f"

    if is_crop_main_expressions_only == False:
        if not os.path.exists( FLAGS.out_to_dir + "/" + dirangle ):
            os.mkdir( FLAGS.out_to_dir + "/" + dirangle )
    else:
        if not os.path.exists( FLAGS.out_to_dir + "/" + dirangle + "_eye" ):
            os.mkdir( FLAGS.out_to_dir + "/" + dirangle + "_eye" )

        if not os.path.exists( FLAGS.out_to_dir + "/" + dirangle + "_mouth" ):            
            os.mkdir( FLAGS.out_to_dir + "/" + dirangle + "_mouth" )

    try:
        #remove image if ratio is not falling in 1:2 in either way 
        if is_crop_main_expressions_only == False and ( im.size[0] > im.size[1]*2 or im.size[1] > im.size[0]*2 ):
            return

        if not FLAGS.merge_and_fit:
            if is_crop_main_expressions_only == True:
                
                xtl = 0; ytl = 0; xbr = 0; ybr = 0; 
                xtl = eye_cords[0] if eye_cords[0] < mouth_cords[0] else mouth_cords[0]
                ytl = eye_cords[1]
                xbr = eye_cords[2] if eye_cords[2] > mouth_cords[2] else mouth_cords[2]
                ybr = mouth_cords[2]

                #create actual face size empty image and paste into it the expressions 
                im = Image.new('RGB', ( int(xbr-xtl), int(ybr-ytl) ) )    

                im.paste( im_expression_eye, ( 0, 0 ) )    

                im.paste( im_expression_mouth, ( int( ( (xbr-xtl) - (mouth_cords[2]-mouth_cords[0]) )/2 ), int( mouth_cords[1]-ytl ) ) )    

                # im.show()
                # cv2.waitKey(0)

            #resize and paste into 36x36
            empty_im = Image.new('RGB', (36, 36))
            resize_to = [0,0]
            if im.size[0] > im.size[1]:
                resize_to[0] = 36
                resize_to[1] = int( im.size[1] / (im.size[0]/resize_to[0]) )
            else:
                resize_to[1] = 36
                resize_to[0] = int( im.size[0] / (im.size[1]/resize_to[1]) )

            im = im.resize( resize_to, Image.ANTIALIAS )

            empty_im.paste( im, ( 0 if im.size[0]==empty_im.size[0] else int( math.floor( (empty_im.size[0]-im.size[0])/2 ) ), 0 if im.size[1]==empty_im.size[1] else int( math.floor( (empty_im.size[1]-im.size[1])/2 ) ) ) )

            #save
            empty_im.save( FLAGS.out_to_dir + "/" + dirangle + "/" + imgname )
            # im.show()
        else:
            is_keep_expression_separate = True 

            if is_keep_expression_separate == False:
                empty_im = Image.new('RGB', (36, 36))

            #resize eye 
            im_expression_eye = im_expression_eye.resize( [27,15], Image.ANTIALIAS )

            #resize mouth 
            im_expression_mouth = im_expression_mouth.resize( [18,13], Image.ANTIALIAS )

            if is_keep_expression_separate == False:
                empty_im.paste( im_expression_eye, ( 5, 0 ) )    

                empty_im.paste( im_expression_mouth, ( 9, 23 ) )    
            else:
                im_expression_eye.save( FLAGS.out_to_dir + "/" + dirangle + "_eye" + "/" + imgname )
                im_expression_mouth.save( FLAGS.out_to_dir + "/" + dirangle + "_mouth" + "/" + imgname )


            if is_keep_expression_separate == False:
                #save
                empty_im.save( FLAGS.out_to_dir + "/" + dirangle + "/" + imgname )
                # im.show()

        #uncomment and enable when necessary 
        # #check if image is of less then or equal to 1KB then is of low quality so remove it 
        # if os.path.getsize( FLAGS.out_to_dir + "/" + dirangle + "/" + imgname ) <= 700:   #1024:    #bytes
        #     os.remove( FLAGS.out_to_dir + "/" + dirangle + "/" + imgname )

        #
        if is_visualize_detected_face == True:
            print( imgpath )
            im_tmp = cv2.imread( imgpath )
            print( im_tmp.size )
            visualize_detected_face( im_tmp, org_csv_row, True )

    except Exception as e:
        print("SKIPPING DUE TO ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR... ERROR...")
        print(e)
        if os.path.exists( FLAGS.out_to_dir + "/" + dirangle + "/" + imgname ):
            os.remove( FLAGS.out_to_dir + "/" + dirangle + "/" + imgname )
        return

    return

def resize( path ):
    items = os.listdir( path )
    file_no = 0
    item_cnt = 0

    for item in items:

        file_no = file_no + 1
        print(item + " file_no " + str(file_no))

        if item == '.DS_Store':
            continue

        if not '.csv' in item:
            continue

        if os.path.isfile(path+item):

            # with open(path+item, newline='') as csvfile:
            #     data = list(csv.reader(csvfile))
            import pandas as pd 
            try:
                data = pd.read_csv(path+item, sep=',')

                # if not FLAGS.verify_against == "":
                #     data_verify = pd.read_csv(FLAGS.verify_path, sep=',')                    
            except Exception as e:
                print("caught error while reading csv "+item+", skipping.")
                print(e)
                continue

            # print(type(data))
            # print(data)

            if FLAGS.out_to_dir:

                prev_rec = {}
                curr_rec = {}
                prev_rec["id"] = ""
                curr_rec["id"] = ""
                prev_rec["lbls"] = []
                curr_rec["lbls"] = []
                prev_row = None

                with open( os.path.join( FLAGS.out_to_dir, item ) , 'wb' ) as file:

                    for index, row in data.iterrows():

                        #
                        if not row[0] == curr_rec["id"]:
                            prev_rec = dict(curr_rec)     #TODO last record will be missed
                            curr_rec["id"] = row[0]
                            curr_rec["lbls"] = []
                            curr_rec["lbls"].append( row[2] )
                        else:
                            prev_row = row
                            curr_rec["lbls"].append( row[2] )
                            continue

                        if prev_rec["id"] == "":
                            continue

                        print(prev_rec["id"])

                        if not FLAGS.verification_val in prev_rec["lbls"]:
                            print("For index "+str(index)+" verification failed against speified value, skipping....")
                            continue

                        # #TODO temp
                        # item_cnt = item_cnt + 1
                        # if item_cnt < 50000 or not item_cnt % 250 == 0:
                        # # if not item_cnt == 9499 and not item_cnt == 9500:
                        #     continue
                        # print( "item_cnt " + str(item_cnt) )

                        #
                        # print(row[2])
                        new_row = do_rotation(prev_row, True if FLAGS.gender_man_lbl in prev_rec["lbls"] else False)
                        # input("Press Enter to continue...")
                        continue


for path in paths:
    resize( path )
            # cv2.imshow("Output", images)
            # cv2.waitKey(0)
