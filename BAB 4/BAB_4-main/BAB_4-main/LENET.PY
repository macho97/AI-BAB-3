from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Input
from keras.models import Sequential

model = Sequential()

model.add(Input(shape=(32, 32, 1)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation="relu"))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=84, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

model.summary()