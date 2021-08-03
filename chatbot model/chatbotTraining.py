##########################################################################################################################
# You should install nltk, numpy, tensorflow package to run this file.
##########################################################################################################################
#!pip install nltk
#!pip install numpy
#!pip install tensorflow

# dictionary
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["hello", "hey", "how are you?", "hi there", "hi", "whats up", "what's up", "how's it going?", "hows it going?", "how you doing?",
                           "how are you doing?", "how have you been?", "how's your day?", "how was your day going so far?", "how are things?", "how's everything?",
                          "yo", "howdy", "how do you do"],
              "responses": ["howdy partner!", "hello", "hi there", "how are you doing?", "greetings!", "how do you do?"],
             },
             {"tag": "diagnosis",
              "patterns": ["what type of cancer do i have?", "what is my exact diagnosis?", "what is my diagnosis?", "what cancer do i have?",
                           "what is the result of diagnosing?", "what is my prognosis?", "what do you know about my diagnosing?"],
              "responses": ["you are lucky because your tumor is benign.", "your tumor has not yet reached an advanced stage."
                           "we found that your tumor is benign."],
             },
             {"tag": "treatment",
              "patterns": ["what are my treatment options?", "which treatment do you recommend?", "what's the goal of my treatment?",
                           "will i be cured?", "is there any possibility that i can be cured?"],
              "responses": ["Surgery is the usual treatment for most brain tumors."],
             },
             {"tag": "financial",
              "patterns": ["how much should i pay?", "how much money is this diagnosis?"],
              "responses": ["it's totally free.", "i don't know either because the diagnosis is totally free"],
             },
             {"tag": "desperate",
              "patterns": ["am i dying?", "my life is over!", "i don't wanna die!", "i'm so sad!", "i feel so bad!"],
              "responses": ["i'm your doctor and i'll do my best to help you!"],
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "g2g", "see ya", "see you", "adios", "cya", "later", "goodbye", "bye bye", "thank you"],
              "responses": ["it was nice speaking to you", "see you later", "speak soon!", "take care!", "stay safe!"]
             }
]}

import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_X = []
doc_Y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_Y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)
for idx, doc in enumerate(doc_X):
    bow = []
    text = (lemmatizer.lemmatize(doc)).lower()
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)

    output_row[classes.index(doc_Y[idx])] = 1
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 1000
# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=1000, verbose=1)

def clean_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens:
    for idx, word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels):
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json):
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents:
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

# running the chatbot
while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print("Doctor:" + result)
