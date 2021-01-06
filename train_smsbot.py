from nltk.stem import WordNetLemmatizer
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import random
from flask import Flask, request
import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Initializing

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, remove duplicates/ignored words
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training Data

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]

    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_p = list(training[:, 0])  # patterns
train_i = list(training[:, 1])  # intents

model = Sequential()
model.add(Dense(128, input_shape=(len(train_p[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_i[0]), activation='softmax'))

# Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_p), np.array(train_i),
                 epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

model.save("my_model")

print("model created")
