import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Dummy data for loss and accuracy (replace with your own training history data)
history = {
    'loss': [
        1.0846422910690308,
        0.43672850728034973,
        0.2767665982246399,
        0.17317882180213928,
        0.1388518065214157,
        0.11060875654220581,
        0.09002680331468582,
        0.061159927397966385,
        0.07992368191480637,
        0.07034513354301453,
    ],
    'accuracy': [
        0.6397651433944702,
        0.8520333766937256,
        0.9052703380584717,
        0.9400659799575806,
        0.9546248912811279,
        0.961613118648529,
        0.9687954783439636,
        0.9793264269828796,
        0.973599910736084,
        0.977530837059021,
    ],
}

# Sidebar for plotting accuracy and loss
st.sidebar.title("Model Performance")

# Plot loss
plt.figure()
plt.plot(history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss over Epochs')
plt.legend()
st.sidebar.pyplot(plt)

# Plot accuracy
plt.figure()
plt.plot(history['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs')
plt.legend()
st.sidebar.pyplot(plt)

# Main title
st.title("AgriTech - Plant Disease Identifier")

# Load the model
model_path = "finalmodel.h5"  # Adjust this path if necessary
loaded_model = tf.keras.models.load_model(model_path, compile=False)

# Plant disease labels
plant_disease_labels = {
    0: 'Bell Pepper Bacterial Spot',
    1: 'Bell Pepper Healthy',
    2: 'Potato Early Blight',
    3: 'Potato Late Blight',
    4: 'Potato Healthy',
    5: 'Tomato Bacterial Spot',
    6: 'Tomato Early Blight',
    7: 'Tomato Late Blight',
    8: 'Tomato Leaf Mold',
    9: 'Tomato Septoria Leaf Spot',
    10: 'Tomato Spider Mites',
    11: 'Tomato Target Spot',
    12: 'Tomato Yellow Leaf Curl Virus',
    13: 'Tomato Mosaic Virus',
    14: 'Tomato Healthy',
}

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class of the image
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])

    # Display the prediction
    st.write("Predicted Index:", predicted_class_index)

    disease=plant_disease_labels[predicted_class_index]
    if 'healthy' in disease:
        st.write(f"This plant is healthy, its predicted class is {plant_disease_labels[predicted_class_index]}")
    else:
        st.write(f"This plant is diseased, its predicted class is {plant_disease_labels[predicted_class_index]}")

else:
    st.write("Please upload an image to proceed.")