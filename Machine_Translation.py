# https://drive.google.com/file/d/1Si7mHErYN6O8ltEyVUtTMb-6jjIJfBGE/view?usp=sharing
# https://www.tensorflow.org/tutorials/text/nmt_with_attention
# https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
import tensorflow as tf
from tensorflow import keras
import numpy as np

# ******************************************************************************
print("------------------------------------------------------------------------")
# Load the data
print("part 1 : Load the data \n")
file_en = open("small_vocab_en.txt","r")
file_fr = open("small_vocab_fr.txt","r")               
raw_data_en = file_en.readlines()
raw_data_fr = file_fr.readlines()

for i in range(3):
    print("En s",i," : ",raw_data_en[i])
    print("Fr s",i," : ",raw_data_fr[i])

print("------------------------------------------------------------------------")
# Add start and end
print("Add start and end \n")

raw_data_fr_in  = ["start " + data for data in raw_data_fr]
raw_data_fr_out = [data + " end" for data in raw_data_fr]
for i in range(3):
    print("Fr s",i," : ",raw_data_fr_in[i])
    print("Fr s",i," : ",raw_data_fr_out[i])

print("------------------------------------------------------------------------") 
# Tokenize
print("Tokenize \n")
def tokenize(x):
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

squ_en, tokenizer_en = tokenize(raw_data_en)

data_fr_in_out = raw_data_fr_in + raw_data_fr_out
squ_fr_in_out, tokenizer_fr = tokenize(data_fr_in_out)

squ_fr_in  = tokenizer_fr.texts_to_sequences(raw_data_fr_in)
squ_fr_out = tokenizer_fr.texts_to_sequences(raw_data_fr_out)

print("Number of distinct words En : ",len(tokenizer_en.word_index))
print("word_index En : \n",tokenizer_en.word_index)
print("-------------------")
print("Number of distinct words Fr : ",len(tokenizer_fr.word_index))
print("word_index Fr : \n",tokenizer_fr.word_index)
print("-------------------")
print("Example : ")
for i in range(1):
    print("En s",i," : ",raw_data_en[i])
    print("squ en : ",squ_en[i],"\n")
    print("Fr in",i," : ",raw_data_fr_in[i])
    print("squ fr in: ",squ_fr_in[i],"\n")
    print("Fr out",i," : ",raw_data_fr_out[i])
    print("squ fr out: ",squ_fr_out[i],"\n")

print("------------------------------------------------------------------------")
# Padding
print("Padding sequence data :\n")
def pad(squ, length=None):
    if length is None:
        length = max([len(sentence) for sentence in squ])
    return keras.preprocessing.sequence.pad_sequences(squ, maxlen=length, padding="post")

squ_en_pad     = pad(squ_en)
squ_fr_pad_in  = pad(squ_fr_in)
squ_fr_pad_out = pad(squ_fr_out)

for i in range(2):
    print("En s",i," : len = ",len(squ_en[i])," : ",squ_en[i])
    print("squ en pad : len = ",len(squ_en_pad[i])," : ",squ_en_pad[i],"\n")
    print("Fr s",i," : len = ",len(squ_fr_in[i])," : ",squ_fr_in[i])
    print("squ fr pad in : len = ",len(squ_fr_pad_in[i])," : ",squ_fr_pad_in[i],"\n")
    
print("------------------------------------------------------------------------")
# This ensures that the beginning of the English sentence will be fed last to the 
# encoder, which is useful because that’s generally the first thing that the decoder needs to translate.
def reverse_squ(squ):
    squ_out = []
    l = len(squ)
    for i in squ:
      elm = i[::-1]
      squ_out.append(elm)
    return np.array(squ_out)

squ_en_input = pad(squ_en)
squ_en_input = reverse_squ(squ_en_input)

batch_size = 5
dataset = tf.data.Dataset.from_tensor_slices((squ_en_input, squ_fr_pad_in, squ_fr_pad_out))
dataset = dataset.shuffle(500).batch(batch_size)

# ******************************************************************************
# ******************************************************************************
# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),tf.zeros([batch_size, self.lstm_size]))

# Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)
        return logits, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),tf.zeros([batch_size, self.lstm_size]))

# ******************************************************************************
vocab_size_en = len(tokenizer_en.word_index) + 1
vocab_size_fr = len(tokenizer_fr.word_index) + 1
len_squ_fr    = len(squ_fr_pad_in[0])

embedding_size = 32
lstm_zize      = 64

optimizer = tf.keras.optimizers.Adam()
epochs    = 20


encoder = Encoder(vocab_size_en, embedding_size, lstm_zize)
decoder = Decoder(vocab_size_fr, embedding_size, lstm_zize)

# ******************************************************************************
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss

def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]

    print("Predict : ")
    print("test_source_text : \n",test_source_text)
    test_source_seq = tokenizer_en.texts_to_sequences([test_source_text])  
    test_source_seq = reverse_squ(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[tokenizer_fr.word_index["start"]]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        out_words.append(tokenizer_fr.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == "end" or len(out_words) >= len_squ_fr:
            break
    print("Output : ")
    print(" ".join(out_words))


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states  = en_outputs[1:]
        de_states  = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits     = de_outputs[0]
        loss       = loss_func(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

# ******************************************************************************
for e in range(epochs):
    en_initial_states = encoder.init_states(batch_size)
    

    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
    print("-----------------------------------")
    print("Epoch : ",e+1," Loss : ",loss.numpy())
    predict()
    print("-----------------------------------")

# ******************************************************************************
while True :
    print("Enter y/n :")
    x = input()
    if x == "n":
        break
    print("text input En : ")
    txt = input()
    predict(txt)
    print("*********")
   
