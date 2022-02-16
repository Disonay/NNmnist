import pickle

import streamlit as st
from keras.datasets import mnist


@st.cache(allow_output_mutation=True)
def load_nn(nn_path):
    with open(nn_path, "rb") as nn_file:
        nn = pickle.load(nn_file)
    return nn


@st.cache
def load_train_test_data():
    return mnist.load_data()
