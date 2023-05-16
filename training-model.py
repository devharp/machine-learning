import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
data = [
    {"text": "I love this product", "sentiment": "positive"},
    {"text": "This movie is amazing", "sentiment": "positive"},
    {"text": "I hate Mondays", "sentiment": "negative"},
    {"text": "The service was terrible", "sentiment": "negative"}
]

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Split the data into input and output
X = df["text"].values
y = df["sentiment"].values

# Tokenize the input text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure equal length
max_sequence_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Convert sentiment labels to numerical values
label_mapping = {"positive": 1, "negative": 0}
y_encoded = [label_mapping[label] for label in y]

# Define the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_padded), np.array(y_encoded), epochs=10)

# Save the model
model.save("models/sentiment_classifier_model.h5")
