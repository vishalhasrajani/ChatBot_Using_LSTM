import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import preprocessing
from PIL import Image
import time

# Load the trained model and tokenizer
model = tf.keras.models.load_model('my_model.keras')
tokenizer = joblib.load('tokenizer.pkl')

# Define a function to convert a sentence to tokens
def str_to_tokens(sentence, maxlen=22):
    tokens_list = []
    words = sentence.lower().split()  # Ensure lowercase and tokenization
    for word in words:
        if word in tokenizer.word_index:
            tokens_list.append(tokenizer.word_index[word])
        else:
            tokens_list.append(tokenizer.word_index.get('UNK', 0))  # Handle OOV words
    # Pad the sequence to the required length (22 in this case)
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen, padding='post')

# Define function to create encoder and decoder models
def make_inference_models(model):
    encoder_inputs = model.input[0]
    encoder_lstm_layer = model.layers[4]
    encoder_lstm_output = encoder_lstm_layer.output
    state_h = encoder_lstm_output[1]
    state_c = encoder_lstm_output[2]
    encoder_states = [state_h, state_c]
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_embedding_layer = model.layers[3]
    decoder_lstm_layer = model.layers[5]
    decoder_dense_layer = model.layers[6]

    decoder_state_input_h = tf.keras.layers.Input(shape=(256,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm_layer(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense_layer(decoder_outputs)

    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

# Create encoder and decoder models
enc_model, dec_model = make_inference_models(model)

# Streamlit interface for user input
st.title("🤖 Chatbot Application")
st.markdown("### Ask me anything, and I'll try to answer! 🌟")

# Create two columns for layout: one for the text and one for the image
col1, col2 = st.columns([3, 1])

# Content in the first column (for user input and response)
with col1:
    st.markdown("### Chat with the Bot! 🗨️")
    
    # Input box for the user question
    user_input = st.text_input("Enter a question:", placeholder="Type your question here... 🧐")

    # Check if there is input to process
    if st.button("Get Answer") and user_input:
        # Display a typing animation for the chatbot
        with st.spinner('🤖 Bot is typing...'):
            time.sleep(2)  # Simulate delay for typing animation

        # Processing the user's input and generating response
        states_values = enc_model.predict(str_to_tokens(user_input))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index.get('start', 1)  # Handle start token
        stop_condition = False
        decoded_translation = ''

        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            
            # Find the corresponding word for the sampled index
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    # Skip appending 'end' token
                    if word != 'end':
                        decoded_translation += ' {}'.format(word)
                    sampled_word = word

            # Stop when the 'end' token is found or when the length of response exceeds 74 words
            if sampled_word == 'end' or len(decoded_translation.split()) > 74:
                stop_condition = True
            
            # Update the target sequence for the next prediction
            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        # Displaying the chatbot's response with emojis and styling
        st.markdown(f"### 🤖 Bot says: {decoded_translation} 🎉")

# Content in the second column (for the image)
with col2:
    img = Image.open("chatbot_image.png")  # Provide path to your image file
    st.image(img, caption='I am your friendly Chatbot! 🤗', use_container_width=False, width=200)

# Add footer with a message (for branding or app purpose)
st.markdown("---")
st.markdown("Made with ❤️ by your friendly Chatbot! 🤖✨")
