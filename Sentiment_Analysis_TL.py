# Sentiment Analysis 2
# Reusing Pretrained Embeddings

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
print("------------------------------------------------------------------------")
# load the dataset
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=("train[:70%]", "train[70%:]", "test"),
    as_supervised=True)

print("------------------------------------------------------------------------")
# Explore the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(5)))
print("train_examples_batch : \n",train_examples_batch)
print("train_labels_batch : \n",train_labels_batch)
print("------------------------------------------------------------------------")
# Create the model
model = keras.Sequential([
hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", dtype=tf.string, input_shape=[], trainable=True),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_data.shuffle(10000).batch(32), epochs=20,validation_data=validation_data.batch(32),
                    verbose=1)
print("------------------------------------------------------------------------")
# plot the learning curves
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(12, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# Evaluate the model
print("Evaluate the model : ")
results = model.evaluate(test_data.batch(32), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

test_examples_batch, test_labels_batch = next(iter(test_data.batch(25000)))
# Confusion Matrix
y_pred = model.predict_classes(test_examples_batch)
class_names = ["0", "1"]
# Compute classification report
from sklearn.metrics import classification_report, confusion_matrix
print("Classification report : \n",classification_report(test_labels_batch, y_pred, target_names = class_names))
# Function to draw confusion matrix
import seaborn as sns
def draw_confusion_matrix(true,preds):
    # Compute confusion matrix
    conf_matx = confusion_matrix(true, preds)
    print("Confusion matrix : \n")
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.show()
    return conf_matx
con_mat = draw_confusion_matrix(test_labels_batch, y_pred)
print("------------------------------------------------------------------------")
