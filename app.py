import pickle
import pandas as pd
import numpy as np
import streamlit as st
import time

model = pickle.load(open("model.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
pad_sequences = pickle.load(open("pad_sequences.pkl", "rb"))

st.title("NEXT WORD PREDICTOR")

text = st.text_input("Please input your text")

if text:
    for i in range(10):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen = 57, padding = "pre")
        pos = np.argmax(model.predict(padded_token_text))

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                st.write(text)
                time.sleep(1)
 