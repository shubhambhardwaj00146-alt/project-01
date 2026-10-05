# STEP 1: Import core Deep Learning libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# STEP 2: Create a simple, readable dataset of messages (1 = Spam, 0 = Safe/Ham)
data = {
    'Message': [
        "Hey, are we still meeting for group study tonight?", # Safe (0)
        "CONGRATULATIONS! You won a free Amazon voucher! Click!", # Spam (1)
        "Can you send me the link for the assignment sheet?", # Safe (0)
        "URGENT: Your bank account is suspended. Verify now.", # Spam (1)
        "Don't forget to submit your lab manual by tomorrow.", # Safe (0)
        "WINNER! Claim your cash prize of Rs 25,000 instantly!" # Spam (1)
    ],
    'Label': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("--- Automated Text Filtering Dataset ---")
print(df)
print("\n---------------------------------------")

# STEP 3: Convert human text into numbers that a neural net can read
# Tokenizer breaks sentences down into individual words
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Message'])
sequences = tokenizer.texts_to_sequences(df['Message'])

# Padding ensures every message vector is the exact same length (10 words)
x = pad_sequences(sequences, maxlen=10, padding='post')
y = np.array(df['Label'])

# STEP 4: Build the Deep Learning Neural Network Architecture
model = Sequential([
    Embedding(input_dim=100, output_dim=8, input_length=10), # Converts words into numerical vectors
    GlobalAveragePooling1D(), # Flattens the text vectors
    Dense(6, activation='relu'), # Hidden Layer for pattern recognition
    Dense(1, activation='sigmoid') # Output Layer (Gives a probability between 0 and 1)
])

# STEP 5: Compile the Neural Network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# STEP 6: Train the Neural Brain
# Epochs = 30 means the network will read through our text dataset 30 times to learn
print("--- Training the Spam Filtering Neural Network ---")
model.fit(x, y, epochs=30, verbose=0)
print("[Success] Deep Learning network trained successfully!\n")

# STEP 7: Practical Field Inference Testing
# Let's test a completely new, unseen incoming message
test_msg = ["You won a lottery cash prize! Click this link!"]
seq_test = tokenizer.texts_to_sequences(test_msg)
test_padded = pad_sequences(seq_test, maxlen=10, padding='post')

# Make the prediction
prediction = model.predict(test_padded, verbose=0)

print("--- Production Inference Evaluation ---")
print(f"Incoming Text: '{test_msg[0]}'")
if prediction > 0.5:
    print(f"⚠️ System Alert: Message classified as SPAM! (Confidence Score: {prediction[0][0]:.2f})")
else:
    print("✅ System Status: Message is SAFE.")
