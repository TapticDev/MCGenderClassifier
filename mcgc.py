import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the pre-trained model
model = tf.keras.models.load_model('classifier.h5')

# Set up the customtkinter window
ctk.set_appearance_mode("Dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

# Create the main window
app = ctk.CTk()
app.title("MCGenderClassifier")
app.geometry("600x370")

# Global variables to hold the image and label
img_label = None
prediction_label = None

# Function to preprocess and predict
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)[0][0]  # Predict using the model

    # Calculate percentage confidence
    percentage_confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    gender = "Male" if prediction > 0.5 else "Female"

    # Display the prediction result
    prediction_label.configure(text=f"Prediction: {gender}\nConfidence: {percentage_confidence}%", text_color="#add8e6")

# Function to upload and display the image
def upload_image():
    global img_label

    file_path = filedialog.askopenfilename()
    if not file_path:
        return  # If no file is selected, return

    # Load and display the image using PIL
    img = Image.open(file_path)
    img = img.resize((300, 300))  # Resize for display
    img = ImageTk.PhotoImage(img)

    # Display the uploaded image in the right frame
    if img_label is None:
        img_label = ctk.CTkLabel(master=image_frame, image=img, text="")
        img_label.image = img
        img_label.pack(padx=10, pady=10)
    else:
        img_label.configure(image=img)
        img_label.image = img

    # Make a prediction on the uploaded image
    predict_image(file_path)

# Create a left frame for the sidebar
sidebar_frame = ctk.CTkFrame(app, width=250, height=600, corner_radius=15)
sidebar_frame.pack(side="left", fill="y", padx=20, pady=20)

# Title in the sidebar
title = ctk.CTkLabel(sidebar_frame, text=" MCGenderClassifier ", font=("Arial", 20, "bold"))
title.pack(pady=30)

# Upload button in the sidebar
upload_button = ctk.CTkButton(sidebar_frame, text="Upload Skin Image", command=upload_image, width=200, height=40, font=("Arial", 16))
upload_button.pack(pady=20)

# Prediction label in the sidebar
prediction_label = ctk.CTkLabel(sidebar_frame, text="Prediction: N/A\nConfidence: N/A", font=("Arial", 18), text_color="#add8e6")
prediction_label.pack(pady=40)

# Create a right frame for the image display
image_frame = ctk.CTkFrame(app, width=600, height=600, corner_radius=15)
image_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

# Run the application
app.mainloop()
