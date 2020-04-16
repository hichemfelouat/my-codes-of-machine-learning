import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from glob import glob
import cv2
from google.colab.patches import cv2_imshow
#*****************************************************************
#*****************************************************************
def read_images(data):
  lst_images = []
  for i in range(len(data)):
    img = cv2.imread(data[i]) 
    img = cv2.resize(img, (128, 128))     
    lst_images.append(img)
  return lst_images
#*****************************************************************
#*****************************************************************  
data_yes = glob("/content/gdrive/My Drive/hichem_data/brain_mri/yes*/*")
data_no  = glob("/content/gdrive/My Drive/hichem_data/brain_mri/no*/*")
lst_imgs_yes  = read_images(data_yes)
lst_imgs_no   = read_images(data_no)

fig = plt.figure()
j = 0
for i in range(6):
  plt.subplot(2,3, i+1)
  plt.tight_layout()
  if i < 3 :    
    plt.imshow(lst_imgs_yes[i], cmap='gray', interpolation='none')
    plt.title("Digit: Yes")
  else:
    plt.imshow(lst_imgs_no[j], cmap='gray', interpolation='none')
    plt.title("Digit: No")
    j = j + 1
  plt.xticks([])
  plt.yticks([])

#*****************************************************************
labels_yes = [1] * len(lst_imgs_yes)
labels_no  = [0] * len(lst_imgs_no)
Y = labels_yes + labels_no
X = lst_imgs_yes + lst_imgs_no

#*****************************************************************
# Scale the pixel intensities down to the [0,1] range by dividing them by 255.0 
# (this also converts them to floats).
import numpy as np
Y = np.asarray(Y)
X = np.asarray(X)
X = X.astype("float32")  
X = X / 255.0

# Split data into train and test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                test_size= 0.20, random_state=100)
print("X_train : ",X_train.shape,"  X_test : ",X_test.shape)

#*****************************************************************
# Creating the model using the Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation="relu", input_shape= (128,128,3)))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())
layer0 = keras.layers.Dense(512, activation="relu",kernel_initializer="he_normal",  
                                kernel_regularizer=keras.regularizers.l2(0.01))
layer1 = keras.layers.Dense(128, activation="relu",kernel_initializer="he_normal",                                                                                        
                                  kernel_regularizer=keras.regularizers.l2(0.01))
layer_output = keras.layers.Dense(1, activation="sigmoid",kernel_initializer="glorot_uniform")

model.add(layer0)
model.add(keras.layers.Dropout(0.2))
model.add(layer1)
model.add(keras.layers.Dropout(0.2))
model.add(layer_output)

# The model’s summary() method displays all the model’s layers
print(model.summary())

#*****************************************************************
# initialize the training data augmentation object
trainAug = keras.preprocessing.image.ImageDataGenerator( rotation_range=15, fill_mode="nearest")
EPOCHS = 200
BS = 32
# Compiling the model
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])

# Validation set
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, 
                                test_size= 0.20, random_state=100)
# Learning rate scheduling
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# Interrupts training when it measures no progress on the validation set
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,  restore_best_weights=True)
# Training the model
history = model.fit_generator( trainAug.flow(X_train, y_train, batch_size=BS), steps_per_epoch=len(X_train) // BS,
                        validation_data=(X_val, y_val), validation_steps=len(y_train) // BS,epochs=EPOCHS,
                        callbacks=[lr_scheduler, early_stopping_cb])

#*****************************************************************
# plot the learning curves
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(12, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# Evaluate the model
model_evaluate = model.evaluate(X_test, y_test)
print("Loss     : ",model_evaluate[0])
print("accuracy : ",model_evaluate[1])

# Confusion Matrix
y_pred = model.predict_classes(X_test)
class_names = ["0", "1"]
# Compute classification report
from sklearn.metrics import classification_report, confusion_matrix
print("Classification report : \n",classification_report(y_test, y_pred, target_names = class_names))
# Function to draw confusion matrix
import seaborn as sns
def draw_confusion_matrix(true,preds):
  # Compute confusion matrix
  conf_matx = confusion_matrix(true, preds)
  print("Confusion matrix : \n")
  sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
  plt.show()
  return conf_matx
con_mat = draw_confusion_matrix(y_test, y_pred)

# Make predictions
# predicted = model.predict(X_new)
#*****************************************************************
