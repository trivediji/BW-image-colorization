import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage import exposure

class EnhancedImageColorizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Colorizer")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.input_image_path = None
        self.output_image_path = None
        self.model_initialized = False
        self.net = None
        self.pts = None
        
        # Load model once during initialization
        self.initialize_model()
        
        # Create GUI elements
        self.create_widgets()
        
    def initialize_model(self):
        try:
            # Model configuration
            prototxt = "colorization_deploy_v2.prototxt"
            weights = "colorization_release_v2.caffemodel"
            points = "pts_in_hull.npy"

            if not all(os.path.exists(f) for f in [prototxt, weights, points]):
                messagebox.showerror("Error", "Model files not found!")
                return

            # Load model and cluster centers
            self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
            self.pts = np.load(points).transpose().reshape(2, 313, 1, 1)
            
            # Configure model layers
            class8 = self.net.getLayerId("class8_ab")
            conv8 = self.net.getLayerId("conv8_313_rh")
            self.net.getLayer(class8).blobs = [self.pts.astype("float32")]
            self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            
            self.model_initialized = True
            
        except Exception as e:
            messagebox.showerror("Error", f"Model initialization failed: {str(e)}")

    def create_widgets(self):
        # Create main container
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display frame
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input image panel
        self.input_panel = tk.Label(self.image_frame, text="Input Image")
        self.input_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        # Output image panel
        self.output_panel = tk.Label(self.image_frame, text="Output Image")
        self.output_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Control panel
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        self.select_btn = tk.Button(
            self.control_frame,
            text="Select Image",
            command=self.select_image,
            width=15
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.colorize_btn = tk.Button(
            self.control_frame,
            text="Colorize",
            command=self.colorize_image,
            width=15,
            state=tk.DISABLED
        )
        self.colorize_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = tk.DoubleVar()
        self.progress_bar = tk.ttk.Progressbar(
            self.control_frame,
            variable=self.progress,
            maximum=100
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.input_image_path = file_path
            self.display_image(self.input_panel, file_path)
            self.colorize_btn.config(state=tk.NORMAL)

    def display_image(self, panel, path):
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            panel.config(image=photo)
            panel.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Image display error: {str(e)}")

    def post_process(self, image):
        try:
            # Step 1: Mild noise reduction
            image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 15)

            # Step 2: Natural contrast adjustment
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Gentle CLAHE with smaller grid
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
            l = clahe.apply(l)
            
            # Merge channels with reduced contrast impact
            lab = cv2.merge((
                cv2.addWeighted(l, 0.7, cv2.blur(l, (5,5)), 0.3, 0),
                a, 
                b
            ))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Step 3: Subtle sharpening with edge mask
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((2,2), np.uint8)
            edge_mask = cv2.dilate(edges, kernel, iterations=1) / 255.0
            
            # Apply sharpening only to edges
            sharpened = cv2.detailEnhance(image, sigma_s=3, sigma_r=0.05)
            image = cv2.addWeighted(image, 1.0 - edge_mask[..., None], 
                                sharpened, edge_mask[..., None], 0)

            # Step 4: Natural color blending
            orig = cv2.imread(self.input_image_path)
            image = cv2.addWeighted(image, 0.8, orig, 0.2, 0)

            # Step 5: Mild tone curve adjustment
            image = exposure.adjust_gamma(image, 0.95)

            return image
        except Exception as e:
            print(f"Post-processing error: {str(e)}")
            return image

    def colorize_image(self):
        if not self.model_initialized:
            messagebox.showerror("Error", "Model not initialized!")
            return

        try:
            # Load and preprocess image
            image = cv2.imread(self.input_image_path)
            original_size = image.shape[:2]
            
            # Convert to float32 and normalize
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            
            # Resize with aspect ratio preservation
            scale_factor = 224 / max(lab.shape[:2])
            new_size = (int(lab.shape[1] * scale_factor), int(lab.shape[0] * scale_factor))
            resized = cv2.resize(lab, new_size, interpolation=cv2.INTER_AREA)
            
            # Extract L channel
            L = cv2.split(resized)[0]
            L -= 50  # Center around 50 for model input
            
            # Colorization
            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
            
            # Resize back to original dimensions
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # Combine channels
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            
            # Post-processing
            colorized = self.post_process(colorized)
            
            # Save and display
            output_path = os.path.splitext(self.input_image_path)[0] + "_colorized.jpg"
            cv2.imwrite(output_path, colorized)
            self.display_image(self.output_panel, output_path)
            
            messagebox.showinfo("Success", f"Image saved as:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Colorization failed: {str(e)}")
        finally:
            self.progress.set(0)

def main():
    root = tk.Tk()
    app = EnhancedImageColorizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()