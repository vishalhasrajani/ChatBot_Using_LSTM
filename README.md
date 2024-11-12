# 🤖 Conversational Chatbot with LSTM | Seq2Seq Model

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square)
![Keras](https://img.shields.io/badge/Built%20With-Keras-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

<img src="templates/chatbot_image.png" alt="Chatbot Workflow" width="70%">

---

## 📝 Introduction

This project develops a conversational chatbot using a Sequence-to-Sequence Long Short-Term Memory (Seq2Seq LSTM) model. Seq2Seq learning enables training of models to transform sequences from one domain to another, which is ideal for tasks such as text generation, speech recognition, and machine translation.

---

## 📄 Project Overview

### Key Elements
- **Language**: Python 3.8
- **Dataset**: [Chatterbot Kaggle English Dataset](https://www.kaggle.com/datasets/kausr25/chatterbot)
- **Libraries Used**: `numpy`, `tensorflow`, `pickle`, `keras`
- **Model**: Seq2Seq LSTM Model
- **API**: Keras Functional API

---

## 🚀 Steps to Build the Chatbot

> **Interactive Features**: Click each section to explore in detail.

<details>
<summary><strong>Step 1: Data Extraction and Preprocessing</strong></summary>

The dataset is derived from Chatterbot's English corpus on Kaggle, containing question-answer pairs across various subjects like food, history, and artificial intelligence.

- **Data Parsing**:
  - Consolidate sentences in responses with multiple parts.
  - Remove unwanted data types from parsing.
  - Add tags to all responses.
  - Create a tokenizer and load the full vocabulary (questions and answers).

- **Data Preparation**:
  - **Encoder Input**: Tokenize and pad questions to a uniform length.
  - **Decoder Input**: Tokenize and pad answers similarly.
  - **Decoder Output**: Tokenize answers, removing the initial element for prediction purposes.

</details>

<details>
<summary><strong>Step 2: Building the Encoder-Decoder Model</strong></summary>

The model employs Embedding, LSTM, and Dense layers for sequence-to-sequence learning.

- **Embedding Layer**: Converts token vectors to dense, fixed-size vectors.
- **LSTM Layer**: Maintains sequential information with Long-Short Term memory cells.
- **Workflow**:
  - The encoder processes the `encoder_input_data` through Embedding and LSTM layers, outputting state vectors (h and c).
  - These state vectors initialize the decoder, which processes `decoder_input_data` to generate sequences.

</details>

<details>
<summary><strong>Step 3: Understanding Long Short-Term Memory (LSTM)</strong></summary>

LSTM networks are a special type of recurrent neural network capable of handling long-term dependencies, useful in language models and conversational AI.

- **Key Components**:
  - **Memory Cells**: Store important information across sequences.
  - **Cell State**: A conveyor belt carrying information throughout the LSTM.
  - **Gates**: Regulate information flow through input, forget, and output gates.

</details>

<details>
<summary><strong>Step 4: Model Training</strong></summary>

The model is trained for 150 epochs using:
- **Optimizer**: RMSprop
- **Loss Function**: Categorical Crossentropy
- **Training Accuracy**: Achieved 96% accuracy after training.

</details>

<details>
<summary><strong>Step 5: Inference Model Setup</strong></summary>

Separate models are used for inference:

1. **Encoder Inference Model**: Takes input questions and generates state vectors (h and c).
2. **Decoder Inference Model**: Accepts encoder state vectors and answer inputs to generate responses.

</details>

<details>
<summary><strong>Step 6: Interacting with the Chatbot</strong></summary>

- **Question to Token Conversion**: Convert input questions into tokenized sequences with padding.
- **Generate State Values**: Use the encoder to obtain state values for each question.
- **Response Generation**:
  - Set initial state values in the decoder LSTM.
  - Start with an initial token and predict the next element.
  - Update the token sequence iteratively until an end tag or max length is reached.

</details>

---

## 🎯 Results

The chatbot successfully generates contextually relevant and coherent responses, demonstrating the Seq2Seq LSTM model's capability for conversational AI tasks.

---

## 🛠️ Tools and Resources

- **Language**: Python 3.8
- **Framework**: Keras (Functional API)
- **Packages**: `numpy`, `tensorflow`, `pickle`

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🎥 Live Demo

> Here’s a GIF demonstrating the chatbot’s response flow.

<img src="templates/chatbot_demo.gif" alt="Chatbot Demo" width="70%">

---

For any questions, suggestions, or contributions, feel free to create an issue or pull request. Enjoy chatting! 💬✨
