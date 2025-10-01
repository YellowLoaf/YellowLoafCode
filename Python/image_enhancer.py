import tkinter as tk
from tkinter import ttk, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from datetime import datetime
import threading
import os

# Import Real-ESRGAN
from realesrgan import RealESRGAN
import torch


class ImageEnhancer:
    def __init__(self):
        # Load AI model (once)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RealESRGAN(self.device, scale=2)
        self.model.load_weights("RealESRGAN_x2.pth", download=True)

        # Main window
        self.window = TkinterDnD.Tk()
        self.window.title("AI Image Enhancer")
        self.window.geometry("1000x700")
        self.window.configure(bg="#2b2b2b")

        # Frame
        self.frame = ttk.Frame(self.window, padding=10)
        self.frame.pack(expand=True, fill="both")

        # Drop label
        self.drop_label = ttk.Label(
            self.frame,
            text="Drag & Drop an image here or click Browse",
            anchor="center",
            font=("Arial", 14),
        )
        self.drop_label.pack(expand=True, pady=20)

        # Image preview
        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(expand=True)

        # Buttons
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(pady=10)

        self.browse_button = ttk.Button(
            self.button_frame, text="Browse Image", command=self.browse_image
        )
        self.browse_button.grid(row=0, column=0, padx=5)

        self.download_button = ttk.Button(
            self.button_frame, text="Download Image", command=self.save_image
        )
        self.download_button.grid(row=0, column=1, padx=5)
        self.download_button.grid_remove()

        # Status bar
        self.status_var = tk.StringVar(value="Waiting for image...")
        self.status_label = ttk.Label(
            self.frame, textvariable=self.status_var, anchor="w", relief="sunken"
        )
        self.status_label.pack(fill="x", side="bottom")

        self.enhanced_image = None

        # Enable drag & drop
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.handle_drop)
        self.window.drop_target_register(DND_FILES)
        self.window.dnd_bind("<<Drop>>", self.handle_drop)

    # ---------------------------
    # Enhancement (AI)
    # ---------------------------
    def enhance_image(self, pil_image: Image.Image) -> Image.Image:
        return self.model.predict(pil_image)

    # ---------------------------
    # Show loading popup
    # ---------------------------
    def show_loading(self, task_func):
        loading_win = tk.Toplevel(self.window)
        loading_win.title("Processing...")
        loading_win.geometry("400x120")
        loading_win.resizable(False, False)
        ttk.Label(loading_win, text="Enhancing image, please wait...").pack(
            pady=10
        )
        progress = ttk.Progressbar(
            loading_win, mode="indeterminate", length=350
        )
        progress.pack(pady=10)
        progress.start(10)

        def run_task():
            try:
                task_func()
            finally:
                progress.stop()
                loading_win.destroy()

        threading.Thread(target=run_task, daemon=True).start()

    # ---------------------------
    # Load & process image
    # ---------------------------
    def process_image(self, file_path):
        try:
            original = Image.open(file_path).convert("RGB")
            self.status_var.set("Enhancing...")

            def task():
                self.enhanced_image = self.enhance_image(original)
                self.display_image(self.enhanced_image)
                self.download_button.grid()
                self.status_var.set(f"Enhanced: {file_path}")

            self.show_loading(task)

        except Exception as e:
            self.status_var.set(f"Error: {e}")

    # ---------------------------
    # Display preview
    # ---------------------------
    def display_image(self, img: Image.Image):
        display = img.copy()
        display.thumbnail((900, 500), Image.LANCZOS)
        photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    # ---------------------------
    # Drag & drop handler
    # ---------------------------
    def handle_drop(self, event):
        file_path = event.data.strip()
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        self.process_image(file_path)

    # ---------------------------
    # Browse button
    # ---------------------------
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            self.process_image(file_path)

    # ---------------------------
    # Save button
    # ---------------------------
    def save_image(self):
        if self.enhanced_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_image_{timestamp}.png"
            self.enhanced_image.save(filename)
            self.status_var.set(f"Saved: {filename}")


# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    app = ImageEnhancer()
    app.window.mainloop()
