import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
import re
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter

data = pd.read_csv('ign.csv', header = 0)
del data['url']

titles = list(data['title'])

feature = list()
title_set = set()
title_dict = dict()
counter = 0
for title in data['title']:
    feature.append(re.sub('[^a-zA-Z ]', '', title).lower())

for title in feature:
    words = title.split(' ')
    for word in words:
        title_set.add(word)
        if not word in title_dict.keys():
            title_dict[word] = counter        
            counter += 1
feature_vector = np.zeros((len(data), len(title_dict.keys())))

for i in range(0, len(feature)):
    words = feature[i].split(' ')
    for word in words:
        count = title.count(word)
        feature_vector[i, int(title_dict[word])] = count
print(feature_vector.shape)

score_phrase_set = set()
for phrase in data['score_phrase']:
    score_phrase_set.add(phrase)

target = list()
for score in data['score_phrase']:
    if score == 'Awful':
        target.append(0)
    elif score == 'Unbearable':
        target.append(1)
    elif score == 'Bad':
        target.append(2)
    elif score == 'Disaster':
        target.append(3)
    elif score == 'Painful':
        target.append(4)
    elif score == 'Great':
        target.append(5)
    elif score == 'Masterpiece':
        target.append(6)
    elif score == 'Okay':
        target.append(7)
    elif score == 'Good':
        target.append(8)
    elif score == 'Amazing':
        target.append(9)
    elif score == 'Mediocre':
        target.append(10)
        
target = np.array(to_categorical(target, nb_classes = 11))

trainX, trainY, testX, testY = train_test_split(feature_vector, target, test_size = 0.3)

# This resets all parameters and variables, leave this here
tf.reset_default_graph()

# Input Layer
net = tflearn.input_data([None, len(title_dict.keys())])

# Hidden Layer
net = tflearn.fully_connected(net, 1000, activation = 'softmax')
net = tflearn.fully_connected(net, 200, activation = 'softmax')
# net = tflearn.fully_connected(net, 20, activation = 'softmax')

# Output Layer
net = tflearn.fully_connected(net, 11, activation = 'softmax')

net = tflearn.regression(net, optimizer = 'sgd', learning_rate = 0.006, loss = 'categorical_crossentropy')

model = tflearn.DNN(net)

model.fit(np.array(feature_vector), np.array(target), validation_set=0.1, show_metric=True, batch_size=32, n_epoch=100)