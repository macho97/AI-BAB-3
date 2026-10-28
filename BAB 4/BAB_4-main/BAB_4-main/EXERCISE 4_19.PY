from transformers import pipeline

# Inisialisasi pipeline sentiment analysis
classifier = pipeline("sentiment-analysis")

# Daftar kalimat uji (gabungan positif dan negatif)
sentences = [
    "I absolutely loved the live concert performance, it was breathtaking!",
    "The album release was postponed again, which is super disappointing.",
    "CORTIS delivered an incredible stage show with amazing energy.",
    "The venue was extremely noisy and the sound quality was terrible.",
]

# Analisis sentiment tiap kalimat
for sentence in sentences:
    result = classifier(sentence)
    print(f"Text: {sentence}")
    print(f"Result: {result}\n")