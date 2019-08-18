# Part 1 - Data Preprocessing
# Artificial neural network

import numpy as np
import matplotlib.pyplot
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode cetegorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])
labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0 )

# Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# part 2 - Make the ANN
# importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(activation= 'relu', input_dim= 11, units = 6, kernel_initializer = 'uniform'))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(activation= 'relu', units = 6, kernel_initializer = 'uniform'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(activation= 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting classifier to the Training set
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Convert probability to binary
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Accuracy
# (1550 + 133) / 2000

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Part 4 - Evaluating, improving  and tuning the ANN
# Run Part 1 first
# Evaluaing the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y = y_train, cv=10, n_jobs=-1, verbose=1)

mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
# Run the part 1 first
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
# results
# batch_size : 32
# epochs : 500
# optimizer : rmsprop
best_accuracy =  grid_search.best_score_
# 0.851

