from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

# Create model
model = Sequential()

# Input Layer & Conv2D
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Compile model
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

print(model.summary())