
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from loading_data import Category

# Load the trained model
model_path = 'models/best_road_model.keras'
model = tf.keras.models.load_model(model_path)

# Get class names from Category enum
class_names = sorted([c.value for c in Category])

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    return predictions[0]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Road Surface Classifier")
        self.geometry("700x750")
        self.config(bg="#f0f0f0")

        # Style configuration
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=6, relief="flat", background="#007bff", foreground="white", font=("Helvetica", 10, "bold"))
        style.map("TButton", background=[('active', '#0056b3')])
        style.configure("TLabel", background="#f0f0f0", foreground="#333", font=("Helvetica", 11))
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Result.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("TProgressbar", thickness=20, troughcolor='#e0e0e0', background='#007bff')

        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Road Surface Condition Classifier", style="Title.TLabel")
        title_label.pack(pady=(0, 20))

        # Select button
        self.select_button = ttk.Button(main_frame, text="Select Image", command=self.select_image, style="TButton")
        self.select_button.pack(pady=10)

        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=10)

        # Results frame
        self.result_frame = ttk.Frame(main_frame, padding="10")
        self.result_frame.pack(pady=10, fill=tk.X)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.show_image(file_path)
            predictions = predict_image(file_path)
            self.display_results(predictions)

    def show_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def display_results(self, predictions):
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        result_title = ttk.Label(self.result_frame, text="Prediction Results", style="Result.TLabel")
        result_title.pack(pady=(10, 5))

        for i, pred in enumerate(predictions):
            frame = ttk.Frame(self.result_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=f"{class_names[i]}", width=15)
            label.pack(side=tk.LEFT, padx=(0, 10))

            progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=300, mode='determinate', value=pred*100, style="TProgressbar")
            progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            percentage_label = ttk.Label(frame, text=f"{pred:.2%}", width=8)
            percentage_label.pack(side=tk.LEFT, padx=(10, 0))

if __name__ == "__main__":
    app = App()
    app.mainloop()
