from sklearn.neural_network import MLPClassifier

# Modifikasi X agar setiap sampel memiliki 3 elemen (3 input)
X = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
]

# Label output sesuai sampel data
y = [0, 1, 1, 1]

# Inisialisasi dan latih MLPClassifier
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)

# Pengujian dengan data uji yang memiliki 3 input
test_data = [[2.0, 2.0, 2.0], [-1.0, -2.0, 0.0]]
print("Hasil Prediksi:", clf.predict(test_data))

# Menampilkan bentuk/dimensi matriks bobot (weight matrix shapes)
print("Bentuk matriks bobot (coefs):", [coef.shape for coef in clf.coefs_])