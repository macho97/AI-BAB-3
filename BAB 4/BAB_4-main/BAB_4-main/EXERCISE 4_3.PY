import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# 1. Load Data
whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7
)

# 2. Build Model dengan 5 Neuron pada Dense Layer
model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),
        Dense(5, activation="relu"),  # Diubah dari 10 menjadi 5 neuron
        Dense(5, activation="relu"),  # Diubah dari 10 menjadi 5 neuron
        Dense(1, activation="sigmoid"),
    ]
)

print(model.summary())

# 3. Compile Model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# 4. Melatih Model & Menghitung Waktu Pelatihan
start_time = time.time()

model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=0,  # Dibuat 0 agar terminal tidak penuh
)

end_time = time.time()
training_time = end_time - start_time

# 5. Evaluasi Performa
results = model.evaluate(X_test, y_test, verbose=0)

print("\n=== HASIL EXERCISE 4.3 (5 NEURONS) ===")
print(f"Training Time : {training_time:.4f} detik")
print(f"Loss          : {results[0]:.4f}")
print(f"Accuracy      : {results[1] * 100:.2f}%")