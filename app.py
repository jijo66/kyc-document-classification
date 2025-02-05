import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt


st.header('Kenyan National KYC Identifier Image Classification Model')
model = load_model('/Users/ggkinuthia/Documents/Masters Course material/AI and Machine Vision/assignment-george/Kenyan KYC classifier project/Image_classification/kenyan_kyc_classifier.keras')
data_cat = ['id', 'passport']
img_height = 180
img_width = 180

# File uploader instead of text input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Commented below code to allow for user to browse and select images on their machine
# image =st.text_input('Enter Image name','passport_sample.jpeg')

image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)



score = tf.nn.softmax(predict)
st.image(uploaded_file, width=200)
st.write('The uploaded image represents ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))

# Additions to visialize
def visualize_predictions(predict, data_cat):
    # Create a bar plot of predictions
    plt.figure(figsize=(8, 5))
    softmax_scores = tf.nn.softmax(predict).numpy()[0]
    
    plt.bar(data_cat, softmax_scores)
    plt.title('Prediction Probabilities')
    plt.xlabel('Categories')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    # Add percentage labels on top of each bar
    for i, v in enumerate(softmax_scores):
        plt.text(i, v, f'{v*100:.2f}%', ha='center', va='bottom')
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Visualize the predictions
visualize_predictions(predict, data_cat)

