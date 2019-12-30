from sklearn.feature_extraction import image
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt 
import matplotlib.image as img
import cv2
import mahotas
import numpy as np

def hu_moments(image):
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  feature = cv2.HuMoments(cv2.moments(image)).flatten()
  return feature

def histogram(image,mask=None):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def haralick_moments(image):
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = image.astype(int)
  haralick = mahotas.features.haralick(image).mean(axis=0)
  return haralick

class ZernikeMoments:
	def __init__(self, radius):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius
 
	def describe(self, image):
		# return the Zernike moments for the image
		return mahotas.features.zernike_moments(image, self.radius)
  
data = fetch_olivetti_faces()
plt.imshow(data.images[0])
patches = image.extract_patches_2d(data.images[0], (2, 2), max_patches=2,random_state=0)
print('Image shape: {}'.format(data.images[0].shape),' Patches shape: {}'.format(patches.shape))
print('Patches = ',patches)

hu_feature = hu_moments(data.images[0])
print("hu_feature : ",hu_feature)

his_feature  = histogram(data.images[0])
print("his_feature : ",his_feature)

har_feature = haralick_moments(data.images[0])
print("har_feature  : ",har_feature )

desc = ZernikeMoments(21)
zer_feature = desc.describe(data.images[0])
print("zer_feature : ",zer_feature)
