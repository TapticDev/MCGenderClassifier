import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import json
import threading

class GenderClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Minecraft Gender Classifier")

        # Create loading screen
        self.loading_screen = tk.Toplevel(self.root)
        self.loading_screen.title("Loading")
        self.loading_label = tk.Label(self.loading_screen, text="Loading model, please wait...")
        self.loading_label.pack(pady=20)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = Progressbar(self.loading_screen, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=20)

        # Start a thread to load the model
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        # Load the trained model
        self.loaded_model = tf.keras.models.load_model('minecraft_gender_classifier.h5')

        # Close the loading screen
        self.loading_screen.destroy()

        # Continue with the main application
        self.setup_main_ui()

    def report_false_positive(self):
        player_name = self.entry.get()

        # Check if a prediction has been made
        if not self.result_label.cget("text"):
            tk.messagebox.showinfo("Error", "Please make a prediction before reporting as false positive.")
            return

        # Determine the predicted gender
        predicted_gender = self.result_label.cget("text").split()[2]

        # Set the destination folder based on the predicted gender
        destination_folder = "female" if predicted_gender.lower() == "male" else "male"

        # Move the skin image to the appropriate folder
        try:
            skin_image_path = f"{destination_folder}/{player_name}.png"
            skin_image = Image.open(BytesIO(requests.get(self.skin_url).content))
            skin_image.save(skin_image_path)

            tk.messagebox.showinfo("Success", f"Skin saved as false positive in the '{destination_folder}' folder.")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save skin: {e}")

    def setup_main_ui(self):
        # Create and set up widgets
        self.label = tk.Label(self.root, text="Enter a player's Minecraft name:")
        self.label.grid(row=0, column=0, pady=10)

        self.entry = tk.Entry(self.root)
        self.entry.grid(row=0, column=1, pady=10)

        self.load_skin_button = tk.Button(self.root, text="Load Skin", command=self.load_and_predict)
        self.load_skin_button.grid(row=0, column=2, pady=10)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.grid(row=1, column=0, columnspan=3, pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=2, column=0, columnspan=3, pady=10)

        # New label for cube render
        self.cube_render_label = tk.Label(self.root)
        self.cube_render_label.grid(row=0, column=3, rowspan=3, padx=10)

        # Report as False Positive button
        self.report_button = tk.Button(self.root, text="Report as False Positive", command=self.report_false_positive)
        self.report_button.grid(row=3, column=3, pady=10)

    def load_and_predict(self):
        player_name = self.entry.get()
        if player_name:
            # Step 1: Obtain the UUID of the Minecraft user
            profile_url = f"https://api.mojang.com/users/profiles/minecraft/{player_name}"
            profile_response = requests.get(profile_url)

            if profile_response.status_code == 200:
                profile_data = json.loads(profile_response.text)
                user_uuid = profile_data.get("id")

                # Step 2: Request information about the player's textures
                if user_uuid:
                    textures_url = f"https://sessionserver.mojang.com/session/minecraft/profile/{user_uuid}"
                    textures_response = requests.get(textures_url)

                    if textures_response.status_code == 200:
                        textures_data = json.loads(base64.b64decode(json.loads(textures_response.text)["properties"][0]["value"]).decode('utf-8'))
                        self.skin_url = textures_data.get("textures", {}).get("SKIN", {}).get("url")

                        if self.skin_url:
                            skin_response = requests.get(self.skin_url)

                            if skin_response.status_code == 200:
                                skin_image = Image.open(BytesIO(skin_response.content))

                                # Convert the image to RGB mode if needed
                                if skin_image.mode != 'RGB':
                                    skin_image = skin_image.convert('RGB')

                                # Display the image
                                img = ImageTk.PhotoImage(skin_image)
                                self.image_label.config(image=img)
                                self.image_label.image = img

                                # Resize the image to match the model's input size
                                skin_image = skin_image.resize((64, 64))
                                skin_array = image.img_to_array(skin_image)
                                skin_array = np.expand_dims(skin_array, axis=0)
                                skin_array /= 255.0

                                # Predict gender
                                prediction = self.loaded_model.predict(skin_array)

                                # Display the result
                                if prediction[0][0] > 0.5:
                                    gender_result = "Male"
                                    confidence = prediction[0][0] * 100
                                else:
                                    gender_result = "Female"
                                    confidence = (1 - prediction[0][0]) * 100

                                self.result_label.config(text=f"Predicted Gender: {gender_result} ({confidence:.2f}%)")

                                # Use threading to fetch and display cube render (body render)
                                threading.Thread(target=self.fetch_and_display_body_render, args=(player_name,), daemon=True).start()
                            else:
                                self.result_label.config(text="Error loading skin. Check player name.")
                        else:
                            self.result_label.config(text="Player does not have a visible skin.")
                    else:
                        self.result_label.config(text="Error fetching player textures.")
                else:
                    self.result_label.config(text="Error fetching player UUID.")
            else:
                self.result_label.config(text="Error fetching player profile.")
        else:
            self.result_label.config(text="Please enter a player's Minecraft name.")

    def fetch_and_display_body_render(self, player_name):
        try:
            # Fetch body rendering from minotar.net
            body_render_url = f"https://minotar.net/armor/body/{player_name}/100.png"
            body_render_image = Image.open(BytesIO(requests.get(body_render_url).content))

            # Display the body rendering
            body_render_img = ImageTk.PhotoImage(body_render_image)
            self.cube_render_label.config(image=body_render_img)
            self.cube_render_label.image = body_render_img
        except Exception as e:
            print(f"Error fetching and displaying body render: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = GenderClassifierGUI(root)
    root.mainloop()
