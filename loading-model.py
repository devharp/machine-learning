import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = [
    "I love this product",
    "This movie is amazing",
    "I hate Mondays",
    "The service was terrible"
]

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)  # X is the input text used during training
X_sequences = tokenizer.texts_to_sequences(X)

# Load the trained model
model = tf.keras.models.load_model("models/sentiment_classifier_model.h5")

# Example text data for prediction
new_text = ["This product is great"]

max_sequence_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)
new_text_sequences = tokenizer.texts_to_sequences(new_text)
new_text_padded = pad_sequences(new_text_sequences, maxlen=max_sequence_length)

predictions = model.predict(new_text_padded)

# Interpret the predictions
sentiment_labels = ["negative", "positive"]
predicted_sentiment = sentiment_labels[int(round(predictions[0][0]))]

print(f"Predicted sentiment: {predicted_sentiment}")
