import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

UwU ="C:/Users/braya/Documents/Python projects/Body Language/df_UwU.csv"
df_UwU = pd.read_csv(UwU)

Sad ="C:/Users/braya/Documents/Python projects/Body Language/df_Sad.csv"
df_Sad = pd.read_csv(Sad)

Happy ="C:/Users/braya/Documents/Python projects/Body Language/df_Happy.csv"
df_Happy = pd.read_csv(Happy)

min_len = np.array([len(df_UwU), len(df_Sad), len(df_Happy)]).min()

clf_cat = {0:'UwU', 1:'Sad', 2:'Happy'}

df = df_UwU[:min_len]
df = pd.concat([df, df_Sad[:min_len], df_Happy[:min_len]], ignore_index=True)
df.reset_index(inplace=True, drop=True)

df['class'] = df['class'].replace(['UwU'], 0)
df['class'] = df['class'].replace(['Sad'], 1)
df['class'] = df['class'].replace(['Happy'], 2)

X = np.array(df.drop(columns='class'))
y = to_categorical(df['class'], num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    shuffle=True,
                                                    test_size=0.3,
                                                    )

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),

    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1, noise_shape=None, seed=42),

    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax'),
])

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=['accuracy'],
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=35,
)

path_model = "C:/Users/braya/Documents/Python projects/Body Language/model.pkl"
with open(path_model, 'wb') as f:
    pickle.dump(model, f)



history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()



