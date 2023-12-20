import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

lemmatizer = WordNetLemmatizer()

# Load your dataset
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] * len(words)
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words:
        if word in word_patterns:
            bag[words.index(word)] = 1

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([item[0] for item in training], dtype=np.float32)
train_y = np.array([item[1] for item in training])

# Model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))  # Adjust dropout rate
model.add(Dense(len(train_y[0]), activation='softmax'))

# Model compilation
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Model training
hist = model.fit(train_x, train_y, epochs=300, batch_size=32, verbose=1)

# Save the model
model.save('chatbotmodel.h5', hist)

print('Done')
