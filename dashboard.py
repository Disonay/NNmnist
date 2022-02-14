import pickle

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from keras.datasets import mnist
from dashboard.plots import plot_example, plot_bar
from sklearn.metrics import accuracy_score


page = st.sidebar.radio("Parts", ["Data", "Recognise handwriting"])
(train_X, train_y), (test_X, test_y) = mnist.load_data()
with open("nn.pickle", "rb") as nn_file:
    nn = pickle.load(nn_file)

if page == "Recognise handwriting":
    image_data = st_canvas(width=300, height=300, stroke_width=38, drawing_mode="freedraw", key="canvas")
    if image_data.image_data is not None:
        digit_gray = 1 - rgb2gray(rgba2rgb(image_data.image_data / 255))
        if st.button("Predict!"):
            st.markdown("Is {}".format(np.argmax(nn.predict(resize(digit_gray, (28, 28)).reshape(1, 784)))))
elif page == "Data":
    st.markdown("## Example of train dataset")
    st.pyplot(plot_example(train_X))
    st.markdown("## Count of each digit")
    st.pyplot(plot_bar(train_y))
    st.markdown("## Accuracy is _{}_".format(accuracy_score(test_y, np.argmax(nn.predict(test_X.reshape(10000, 784) / 255), axis=1))))

