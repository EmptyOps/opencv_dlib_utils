import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu',flip_input=False)

input = io.imread('../test/__assets/img1.jpeg')
preds = fa.get_landmarks(input)[-1]
# preds = fa.get_landmarks_from_directory('../test/assets/')[-1]

#TODO: Make this nice
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input)
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


print("ax Print Her --",type(preds))
print("Here_size",len(preds))
print( preds )

# with open(preds.out_to_csv_file , 'wb' ) as file:

#             for item in items:

#                 print(item)

#                 if item == '.DS_Store':
#                     continue

#                 if os.path.isfile(path+item):
                
#                     line = "\""+item+"\";"

#                     # load the input image, resize it, and convert it to grayscale
#                     images = cv2.imread(path+item)
#                     images = imutils.resize(images, width=500)
#                     gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) 
#                     f, e = os.path.splitext(path+item)
#                     rects = detector(gray, 1)

#                     for (i, rect) in enumerate(rects):
#                         # determine the facial landmarks for the face region, then
#                         # convert the facial landmark (x, y)-coordinates to a NumPy
#                         # array
#                         shape = predictor(gray, rect)
#                         shape = face_utils.shape_to_np(shape)

#                         for (x, y) in shape:
                                
#                             line = line + ";\""+str(x)+"-"+str(y)+"\""

#                         file.write(line.encode())
#                         file.write('\n'.encode())

plt.show()
