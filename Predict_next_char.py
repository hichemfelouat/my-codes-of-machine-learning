# Predict next char 
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Download the Shakespeare dataset
shakespeare_url = "https://homl.info/shakespeare" # shortcut URL
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()
print("------------------------------------------------------------------------")
# length of text is the number of characters in it
print ("Length of text: ",len(shakespeare_text)," characters.")
# Take a look at the first 250 characters in text
print(shakespeare_text[:250])
print("shakespeare_text :\n")
# print(shakespeare_text)
print("------------------------------------------------------------------------")
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

dataset_size = tokenizer.document_count
print("Total number of documents : \n",dataset_size)
max_id = len(tokenizer.word_index) 
print("Number of distinct characters/words : \n",max_id)
print("word_index : \n",tokenizer.word_index)
# print("word_counts : \n",tokenizer.word_counts)
# print("word_docs : \n",tokenizer.word_docs)
print("------------------------------------------------------------------------")
# Let’s encode the full text so each character is represented by its ID
# 0 to 38
encoded1 = tokenizer.texts_to_sequences(shakespeare_text)
encoded = []
print("encoded : ")
for i in range(len(encoded1)):
    encoded.append(encoded1[i][0]-1)
# Create tensor slices
print("Create tensor slices : ")
dataset = tf.data.Dataset.from_tensor_slices(encoded)
for elem in dataset.take(5):
    print(elem.numpy())

"""
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
for elem in dataset:
    print(elem.numpy())
8
3
0
8
2
1
"""
print("------------------------------------------------------------------------")
# The batch method lets us easily convert these individual characters to sequences of the desired size.
seq_length = 100 # target = input shifted 1 character ahead
sequences = dataset.batch(seq_length+1, drop_remainder=True) 
print("The batch method : ")
for item in sequences.take(3):
    print("Len item : ",len(item))
    print(item.numpy())

print("------------------------------------------------------------------------")
# For each sequence, duplicate and shift it to form the input and target text 
# by using the map method to apply a simple function to each batch:
def split_input_target(seq):
    input_text  = seq[:-1]
    target_text = seq[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
print("X and Y :")
for input_example, target_example in  dataset.take(2):
    print ("Input data  : \n",input_example)
    print ("Target data : ",target_example)

print("------------------------------------------------------------------------")
# Create training batches
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print("Dataset : \n",dataset)
print("------------------------------------------------------------------------")
# ******************************************************************************
# ******************************************************************************

# Create The Model

def create_model(vocab_size, embedding_dim, batch_size):
    model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]),
 
   keras.layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, 
                    stateful=True, recurrent_initializer="glorot_uniform"),
    keras.layers.GRU(39, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, 
                    stateful=True, recurrent_initializer="glorot_uniform"),
   
    keras.layers.Dense(max_id, activation="softmax") ])
    return model

model_train = create_model(max_id, 64, BATCH_SIZE)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model_train.compile(optimizer="adam", loss=loss)
history = model_train.fit(dataset, epochs=10)


model_predict = create_model(max_id, 64, 1)
weights = model_train.get_weights()
model_predict.set_weights(weights)
# ******************************************************************************
# ******************************************************************************
print("------------------------------------------------------------------------")
def listToString(lst):  
    str1 = ""   
    for ele in lst:  
        str1 += ele[0]      
    return str1 

def generate_text(model, tokenizer, nbr, tmp, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = nbr

    # Converting our start string to numbers (vectorizing)
    # input_eval = [char2idx[s] for s in start_string]
    encoded_generate_text = tokenizer.texts_to_sequences(start_string)
    input_eval = []
    for i in range(len(encoded_generate_text)):
        input_eval.append(encoded_generate_text[i][0]-1)
    input_eval = tf.expand_dims(input_eval, 0)
    
    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = tmp

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        #predictions = model(input_eval, batch_size = 1)
        predictions = model.predict(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # text_generated.append(idx2char[predicted_id])
        # text_generated.append(tokenizer.sequences_to_texts([predicted_id]))
        # print("predicted_id = ",predicted_id)
        # print("predicted_id char = ",tokenizer.sequences_to_texts([[predicted_id]]))
        text_generated.append(tokenizer.sequences_to_texts([[predicted_id]]))
            

    #return (start_string + " ".join(text_generated))
    return start_string + listToString(text_generated)
    
# ******************************************************************************    
while True :
    print("Enter y/n :")
    x = input()
    if x == "n":
        break
    print("Starting string : ")
    sts = input()
    print("Number of characters to generate : ")
    nbr = input()
    nbr= int(nbr)
    print("Temperature : ")
    tmp = input()
    tmp = float(tmp)
    print("Predicted characters :")
    print(generate_text(model_predict, tokenizer, nbr, tmp, start_string= sts))
    print("++++++++++++++++++++++++")

print("------------------------------------------------------------------------")
# ******************************************************************************
