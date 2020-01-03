from sklearn import datasets 
import cv2
import numpy as np
from silx.opencl import sift
import matplotlib.pyplot as plt
from PIL import Image

def Rot(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

data = datasets.fetch_olivetti_faces()
img1 = data.images[0]
#img2 = Rot(data.images[0], -90)
img2 = cv2.rotate(data.images[0], cv2.ROTATE_90_CLOCKWISE)

sift_ocl = sift.SiftPlan(template=img, devicetype="GPU")
keypoints_1 = sift_ocl.keypoints(img1)
keypoints_2 = sift_ocl.keypoints(img2)

figure, ax = plt.subplots(1, 2,)
ax[0].imshow(img1, cmap='gray')
ax[0].plot(keypoints_1[:].x, keypoints_1[:].y,".r")
ax[1].imshow(img2, cmap='gray')
ax[1].plot(keypoints_2[:].x, keypoints_2[:].y,".r")

mp = sift.MatchPlan()
matching_keypoints = mp(keypoints_1, keypoints_2)
print("Number of Keypoints - image 1 :", 
      keypoints_1.size," - image 2 : ",keypoints_2.size, 
      " - Matching keypoints : ", matching_keypoints.shape[0])

img3 = cv2.hconcat([img1, img2])
fig, ax = plt.subplots()
ax.imshow(img3, cmap='gray')
ax.axis('off')
ax.plot(matching_keypoints[:,0].x, matching_keypoints[:,0].y,'.r')
ax.plot(matching_keypoints[:,1].x+img1.shape[1], matching_keypoints[:,1].y,'.r')
for m in matching_keypoints:
    ax.arrow(m[0].x, m[0].y, m[1].x-m[0].x+img1.shape[1], m[1].y-m[0].y)
fig.show()
