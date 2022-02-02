from prepare_data import load_data,generate_y,partition, y_nextframe
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_path = "data/pn157_combined.h5"
#data_path = "combination_all_pn157.h5"

target_path = ""
square = (250,650,0,400)
data,dates = load_data(data_path,N = 1000,concat = 13, square = square, downsampling_rate = 8, overlap = 0)

data, y = generate_y(data,dates)
#data, y = y_nextframe(data,dates)

Xtrain,Xtest,Ytrain,Ytest = partition(data, y, 0.8)
print("Xtrain shape: ",Xtrain.shape)
print("Xtest shape: ",Xtest.shape)
print("Ytrain shape: ",Ytrain.shape)
print("Ytest shape: ",Ytest.shape)
shape = Xtrain.shape
mid_x = (shape[2])//2
mid_y = (shape[3])//2

# ------------ NETWORK -----------------


print("INPUT, ", *Xtrain.shape[2:])
inp = layers.Input(shape=Xtrain.shape[1:])
output_shape = Ytrain.shape[1]
print("output_shape ",output_shape)
# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(200, activation="relu",activity_regularizer=tf.keras.regularizers.l2(2e-6))(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(100, activation="relu",activity_regularizer=tf.keras.regularizers.l2(2e-6))(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(output_shape, activation="relu",activity_regularizer=tf.keras.regularizers.l2(2e-6))(x)


# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)


model.summary()

# Define modifiable training hyperparameters.
epochs = 5
batch_size = 5
# Fit the model to the training data.
model.fit(
    Xtrain,
    Ytrain,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(Xtest, Ytest),
    callbacks=[early_stopping, reduce_lr],
)
'''
# Select a random example from the validation dataset.

example = Ytest[0]


# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()

'''
