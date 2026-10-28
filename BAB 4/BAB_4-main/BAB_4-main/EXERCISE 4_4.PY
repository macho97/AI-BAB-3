from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Input
from keras.models import Sequential

model = Sequential()

# Layer Input & Conv1 (diubah dari 6 menjadi 12 filter)
model.add(Input(shape=(32, 32, 1)))
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation="relu"))
model.add(AveragePooling2D(pool_size=(2, 2)))

# Conv2 (diubah dari 16 menjadi 32 filter)
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(AveragePooling2D(pool_size=(2, 2)))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=84, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

model.summary()