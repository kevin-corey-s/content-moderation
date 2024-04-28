import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

class ImageClassifier(TkinterDnD.Tk):
    def __init__(self, model_path):
        super().__init__()
        self.model = load_model(model_path)
        self.class_labels = ['non-explicit', 'knife', 'pistol', 'rifle']
        self.title("Image Classifier")
        self.geometry("600x400")

        self.label = tk.Label(self, text="Drag and drop an image here", padx=20, pady=20)
        self.label.pack(expand=True, fill=tk.BOTH)

        # Setup drag and drop listening
        self.label.drop_target_register(DND_FILES)
        self.label.dnd_bind('<<Drop>>', self.drop)

    def drop(self, event):
        file_path = event.data
        print("Original path:", file_path)  # Debug: print the original path
        if file_path.startswith('file://'):
            file_path = file_path[7:]  # Remove the 'file://' prefix
        file_path = os.path.normpath(file_path)
        print("Processed path:", file_path)  # Debug: print the processed path
        self.classify_image(file_path)


    def classify_image(self, file_path):
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist: " + file_path)
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image: " + file_path)
            return
        orig = image.copy()

        # Preprocess the image
        image = cv2.resize(image, (256, 256))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Predict the class
        predictions = self.model.predict(image)[0]
        explicit_indices = [1, 2, 3]  # assuming these indices correspond to 'knife', 'pistol', 'rifle'
        explicit_probability = sum(predictions[index] for index in explicit_indices)
        # Create a detailed label text with all class probabilities
        label_text = "Classification Results:\n"
        for i, prob in enumerate(predictions):
            label_text += f"{self.class_labels[i]}: {prob * 100:.2f}%\n"
        label_text += f"\nTotal Explicit Probability: {explicit_probability * 100:.2f}%"

        self.label.config(text=label_text)

        # Display the original image in the GUI
        img = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img.resize((250, 250), Image.Resampling.LANCZOS))
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        # Show result in a message box
        messagebox.showinfo("Classification Result", label_text)

if __name__ == "__main__":
    app = ImageClassifier(model_path="all.h5")  # Ensure the model path is correct
    app.mainloop()