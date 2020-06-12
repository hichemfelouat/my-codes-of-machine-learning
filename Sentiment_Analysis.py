# Sentiment Analysis 
import tensorflow as tf
from tensorflow import keras
import numpy as np


print("------------------------------------------------------------------------")
# load the dataset 
print("------------------------------------------------------------------------")
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)

for i in range(5):
    print("-----------------------------------")
    print("Exp : ",i," len : ",len(X_train[i])," lable : ",y_train[i])
    print(X_train[i])
print("------------------------------------------------------------------------")
X = np.concatenate((X_train, X_test), axis=0)

print("Padding sequence data :")
max_words = 500
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_words)
X_test  = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_words)
print("X_train shape : ",X_train.shape," X_test shape : ",X_test.shape)
print("------------------------------------------------------------------------")
# Create the model
model = keras.models.Sequential([
    keras.layers.Embedding(top_words, 128, input_shape= [max_words]),
     keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=1, batch_size=64)
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

# ******************************************************************************  
print("------------------------------------------------------------------------") 
vocabulary = np.unique(np.hstack(X))

def reviewToseq(review, vocabulary, max_words):
    word_to_id = keras.datasets.imdb.get_word_index()
    lstwords = keras.preprocessing.text.text_to_word_sequence(review)
    seq = []
    for word in lstwords:
        id = word_to_id.get(word)
        if id in vocabulary :       
            seq.append(id)
        else:
            seq.append(0)    
    seq = keras.preprocessing.sequence.pad_sequences([seq], maxlen= max_words)
    return seq
  
while True :
    print("Enter y/n :")
    x = input()
    if x == "n":
        break
    print("Review : ")
    review = input()
    seq = reviewToseq(review, vocabulary, max_words)
    #print("seq : \n",seq)
    
    cls = model.predict_classes(seq)
    print("Class : ", cls[0])
    
    print("++++++++++++++++++++++++")

print("------------------------------------------------------------------------")
# ******************************************************************************
