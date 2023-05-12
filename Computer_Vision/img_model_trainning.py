import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

path ="C:/Users/braya/Documents/Python projects/Body Language/video_train_balanced.csv"
df = pd.read_csv(path)

clf_cat = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
for key in clf_cat:
    df[df.columns[0]] = df[df.columns[0]].replace([clf_cat[key]], key)

X = np.array(df.drop(columns=df.columns[0]))
y = to_categorical(df[df.columns[0]], num_classes=len(clf_cat))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    shuffle=True,
                                                    test_size=0.4,
                                                    )

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),

    layers.BatchNormalization(),
    layers.Dropout(0.5, noise_shape=None, seed=42),

    # First Layer
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5, noise_shape=None, seed=42),

    # Second Layer
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5, noise_shape=None, seed=42),

    # Third Layer
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5, noise_shape=None, seed=42),

    # Head
    layers.BatchNormalization(),
    layers.Dense(len(clf_cat), activation='softmax'),
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
    epochs=50000,
)

path_model = "C:/Users/braya/Documents/Python projects/Body Language/vid_model_50k"

tf.keras.models.save_model(
    model=model,
    filepath=path_model,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()




