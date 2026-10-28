from numpy import dot, array, exp, random

# Set up parameters ===============================================
# Input: [0,0], [1,1], [1,0], [0,1]
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])

# Target Output logika OR:
outputs = array([[0, 1, 1, 1]]).T

# Inisialisasi bobot awal secara acak
random.seed(1)
weights = 2 * random.random((2, 1)) - 1

# Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

# Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    # Updating weights menggunakan derivative dari sigmoid activation function
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron =================================================
# Menguji semua kombinasi input untuk memastikan logika OR berjalan dengan benar
print("Hasil Pengujian Logika OR:")
for x in inputs:
    prediksi = neuron(x, weights)[0]
    print(f"Input: {x} -> Output: {prediksi:.4f} (Dibulatkan: {round(prediksi)})")