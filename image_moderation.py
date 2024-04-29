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
        self.class_labels = ['non-explicit', 'knife', 'pistol', 'rifle'] #the classes that we used
        self.title("Image Classifier") #title for the gui
        self.geometry("600x400") #set dimenions of the gui component

        self.label = tk.Label(self, text="Drag and drop an image here", padx=20, pady=20) #display the gui for drag and drop and display the correct test
        self.label.pack(expand=True, fill=tk.BOTH)
        self.label.drop_target_register(DND_FILES) #setup drag and rop listening
        self.label.dnd_bind('<<Drop>>', self.drop) 

    def drop(self, event):
        file_path = event.data
        if file_path.startswith('file://'): #if it start with 'file://'
            file_path = file_path[7:]  # removes the 'file://' prefix
        file_path = os.path.normpath(file_path)
        self.classify_image(file_path)


    def classify_image(self, file_path):
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist: " + file_path)
            return

        image = cv2.imread(file_path) #read in data from the image
        if image is None:
            messagebox.showerror("Error", "Failed to load image: " + file_path)
            return
        orig = image.copy() 

        image = cv2.resize(image, (256, 256)) #resize the image to 356x256
        image = image.astype("float") / 255.0 #parse type to e used
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)[0] #predict the classes of the images
        explicit_indices = [1, 2, 3]  #indices correspond to 'knife', 'pistol', 'rifle'
        explicit_probability = sum(predictions[index] for index in explicit_indices) #get the probabilities for each
        label_text = "Classification Results:\n" #print that this is the classification results.
        for i, prob in enumerate(predictions):
            label_text += f"{self.class_labels[i]}: {prob * 100:.2f}%\n"
        label_text += f"\nTotal Explicit Probability: {explicit_probability * 100:.2f}%" #add all of the expliit proabbilities together and output it

        self.label.config(text=label_text)

        img = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)) #display the image in the gui.
        imgtk = ImageTk.PhotoImage(image=img.resize((250, 250), Image.Resampling.LANCZOS))
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        messagebox.showinfo("Classification Result", label_text) #show the results in the messagebox

if __name__ == "__main__":
    app = ImageClassifier(model_path="all.h5")  #model path, you ma need to ensure it's correct.
    app.mainloop()