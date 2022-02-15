# -*- encoding: utf-8 -*-

import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas

from dashboard.load_data import load_nn, load_train_test_data
from dashboard.plots import plot_bar, plot_example
from my_nn.utils import image_preprocessing

(train_X, train_y), (test_X, test_y) = load_train_test_data()
nn = load_nn("nn.pickle")

page = st.sidebar.radio("Parts", ["Data", "Recognise handwriting"])

if page == "Recognise handwriting":
    image_data = st_canvas(
        width=300, height=300, stroke_width=28, drawing_mode="freedraw", key="canvas"
    )
    if image_data.image_data is not None:
        if st.button("Predict!"):
            st.markdown(
                "Is {}".format(
                    np.argmax(
                        nn.predict(image_preprocessing(image_data.image_data).reshape(1, 784))
                    )
                )
            )
elif page == "Data":
    st.markdown("## Example of train dataset")
    st.pyplot(plot_example(train_X))
    st.markdown("## Count of each digit")
    st.pyplot(plot_bar(train_y))
    st.markdown(
        "## Accuracy is _{}_".format(
            accuracy_score(test_y, np.argmax(nn.predict(test_X.reshape(10000, 784) / 255), axis=1))
        )
    )
