# BW-image-colorization
Image Colorizer Using Caffe Model
## **Image Colorizer Using Caffe Model**

### **Overview**  
This project implements an AI-powered **image colorization system** that converts black-and-white images into **realistically colorized** versions using **deep learning techniques**. The model is based on a **pre-trained Caffe neural network** and is integrated with a **Graphical User Interface (GUI)** using Tkinter for ease of use.  

### **Features**  
✅ **Automatic Image Colorization:** Converts grayscale images into vibrant colored images.  
✅ **Deep Learning-Based Approach:** Uses a **Caffe model** trained on large datasets.  
✅ **User-Friendly GUI:** Tkinter-based interface for easy image selection and processing.  
✅ **High-Quality Post-Processing:** Improves results with sharpening, contrast adjustment, and noise reduction.  
✅ **Fast and Efficient:** Provides **one-click** image colorization.  

### **How It Works?**  
1. **Load a grayscale image** through the GUI.  
2. The image is converted into **LAB color space** for processing.  
3. The **Caffe model** predicts **A and B color channels** while preserving the **Luminance channel**.  
4. The output is converted back to **RGB format** for a natural look.  
5. **Post-processing enhancements** improve image clarity.  
6. The final colorized image is displayed and can be saved.  

### **System Architecture**  
📌 **Input Module:** Accepts grayscale images.  
📌 **Preprocessing:** Converts images to LAB color space.  
📌 **Deep Learning Model:** Predicts missing colors using the Caffe neural network.  
📌 **Post-Processing:** Enhances quality with noise reduction and sharpening.  
📌 **Output Module:** Displays and saves the colorized image.  

### **Technologies Used**  
- **Python** 🐍  
- **OpenCV** 👁️ (for image processing)  
- **Caffe Model** 🧠 (pre-trained deep learning model)  
- **Tkinter** 🖥️ (for GUI implementation)  

### **Results**  
✔️ **Accurate colorization** of black-and-white images.  
✔️ **Easy-to-use GUI** with one-click processing.  
✔️ **Performance evaluated using SSIM and PSNR** for image quality assessment.  

### **Future Enhancements**  
🚀 Train custom models for **higher accuracy** on specific datasets.  
🚀 Improve **real-time processing speed**.  
🚀 Integrate **adaptive deep learning** for more precise color prediction.  

### **Installation & Usage**  
1. Clone the repository:  
```bash
git clone https://github.com/yourusername/Image-Colorizer.git
cd Image-Colorizer
```
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
3. Run the application:  
```bash
python main.py
```
4. Upload a grayscale image and colorize it with **one click!** 🎨  

---

### **Contributors**  
👨‍💻 **Om Kumar Trivedi** (omkr2204@gmail.com)  
👨‍💻 **Aditya** (aditya814sinha@gmail.com)  

Feel free to contribute and improve this project! 🤝🚀  

**⭐ Star this repository if you found it useful!** 🌟

