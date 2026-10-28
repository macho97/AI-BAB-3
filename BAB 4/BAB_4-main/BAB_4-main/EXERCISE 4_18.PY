import nltk

# Download resource pendukung
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")

# Teks baru untuk analisis 
sentence = """
CORTIS is an alternative pop-rock band based in Surabaya. 
They released their new single under a local music label, and the band leader, Alex, announced an upcoming tour in Jakarta.
"""

# 1. Tokenization
tokens = nltk.word_tokenize(sentence)
print("--- TOKENS ---")
print(tokens)

# 2. POS Tagging
tagged = nltk.pos_tag(tokens)
print("\n--- POS TAGS ---")
print(tagged)

# 3. Named Entity Recognition (NER)
entities = nltk.chunk.ne_chunk(tagged)
print("\n--- ENTITIES ---")
print(entities)