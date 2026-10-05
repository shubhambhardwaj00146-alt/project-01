# project-01
# PhishShield-DL: Deep Learning Spam SMS and Threat Filter

## 🚀 Project Context & Overview
Developed as a B.Tech CSAIME Semester 1 project, this repository implements a Sequential Artificial Neural Network (ANN) framework to automatically classify textual short message services (SMS) into Safe (Ham) or Dangerous (Spam) pools. Designing autonomous natural language threat filters mirrors endpoint security architectures deployed by communication giants like Truecaller, WhatsApp, and Gmail to intercept phishing vectors.

## 🛠️ Software Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Language Stack:** Python 3
* **Data Engineering Blocks:** NumPy, Pandas

## 📊 Feature Matrix Architecture
The model processes text vectors based on natural language architecture:
* **Message:** Raw textual text sequence metrics (String/Text vectors)
* **Label:** Categorical binary targets (0: Safe/Ham, 1: Malicious Spam)
* **Pre-processing:** Tokenization via text-to-sequence mapping followed by structural 10-word vector padding constraints.

## 🧠 Neural Network Layer Architecture
* **Embedding Layer:** Maps tokenized word indices into structural multi-dimensional dense vector spaces.
* **Global Average Pooling:** Downsamples text patterns to optimize operational processing speed.
* **Dense Hidden Layer:** Implements a non-linear 'ReLU' activation function for linguistic trend detection.
* **Dense Output Layer:** Deploys a mathematical 'Sigmoid' activation function to compute a definitive classification probability metric scaled precisely between 0.0 and 1.0.
