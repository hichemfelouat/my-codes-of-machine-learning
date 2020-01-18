# Load the data
# dataset : https://drive.google.com/open?id=1n9TAA8zZlKi0aUSVJptdXME07IbNlzfO
En, Fr = [], []
fEn = open("small_vocab_en.txt","r")
fFr = open("small_vocab_fr.txt","r")
for l in fEn:
  En.append(l)
for l in fFr:
  Fr.append(l)
print("part 1 : Load the data *******************************")
for i in range(2):
  print('En s {}:  {}'.format(i + 1, En[i]))
  print('Fr s {}:  {}'.format(i + 1, Fr[i]))

# Tokenize
from keras.preprocessing.text import Tokenizer
def tokenize(x):
  x_tk = Tokenizer(char_level = False)
  x_tk.fit_on_texts(x)
  return x_tk.texts_to_sequences(x), x_tk

Entext_tokenized, Entext_tokenizer = tokenize(En)
Frtext_tokenized, Frtext_tokenizer = tokenize(Fr)
print("part 2 : Tokenize *******************************")
print("En Vocabulary  : ", Entext_tokenizer.word_index)
print("Input 1: ", En[0])
print("Output1: ", Entext_tokenized[0])
print("")
print("Fn Vocabulary  : ", Frtext_tokenizer.word_index)
print("Input 2: ", Fr[0])
print("Output2: ", Frtext_tokenized[0])

# Padding
print("")
print("part 3 : Padding *******************************")
from keras.preprocessing.sequence import pad_sequences
def pad(x, length=None):
  if length is None:
    length = max([len(sentence) for sentence in x])
  return pad_sequences(x, maxlen = length, padding = 'post')

Entest_pad = pad(Entext_tokenized)
Frtest_pad = pad(Frtext_tokenized)
print("En 0 :", Entext_tokenized[0] , " Len = ", len(Entext_tokenized[0]))
print("En 1 :", Entest_pad[0], " Max Len = ", len(Entest_pad[0]))
print("")
print("Fr 0 :", Entext_tokenized[0] , " Len = ", len(Entext_tokenized[0]))
print("Fr 1 :", Entest_pad[0], " Max Len = ", len(Frtest_pad[0]))

print("")
print("part 4 : RNN *******************************")
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.models import Model
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import numpy as np

Frtest_pad = Frtest_pad.reshape(*Frtest_pad.shape, 1)

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  learning_rate = 1e-3
  input_seq = Input(input_shape[1:])
  rnn = GRU(64, return_sequences = True)(input_seq)
  logits = TimeDistributed(Dense(french_vocab_size))(rnn)
  model = Model(input_seq, Activation('softmax')(logits))
  model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
  return model

max_french_sequence_length = 21
english_vocab_size = 199
french_vocab_size  = 344
tmp_x = pad(Entest_pad, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, Frtest_pad.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)
simple_rnn_model.fit(tmp_x, Frtest_pad, batch_size=1024, epochs=50, validation_split=0.2)

def logits_to_text(logits, tokenizer):
  index_to_words = {id: word for word, id in tokenizer.word_index.items()}
  index_to_words[0] = '<PAD>'
  return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

#print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], Frtext_tokenizer))
St0 = "I am going to new jersey"
St1 = [[96, 197, 126, 81, 17, 23]]
print("St0 = ",St.strip())
print(St1)
St2 = pad_sequences(St1,maxlen = 21, padding = 'post')
St3 = St2.reshape((-1, Frtest_pad.shape[-2], 1))
print(logits_to_text(simple_rnn_model.predict(St3)[0], Frtext_tokenizer))
