import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.special import gamma, factorial
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, Scale, Label, HORIZONTAL, Button

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.asarray(img) / 255.0
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img

def apply_gaussian_filter(img, sigma):
    return gaussian_filter(img, sigma=sigma)

def fractional_derivative(img, alpha, axis, kernel_size):
    half = kernel_size // 2
    kernel = np.zeros(kernel_size)
    for k in range(-half, half + 1):
        coeff = gamma(alpha + 1) / (factorial(abs(k)) * gamma(alpha - abs(k) + 1))
        kernel[k + half] = ((-1) ** k) * coeff
    kernel = kernel.reshape(1, -1) if axis == 0 else kernel.reshape(-1, 1)
    return convolve(img, kernel, mode='nearest')

def compute_gradients(img, alpha, kernel_size):
    fx = fractional_derivative(img, alpha, 0, kernel_size)
    fy = fractional_derivative(img, alpha, 1, kernel_size)
    G = np.sqrt(fx**2 + fy**2)
    theta = np.arctan2(fy, fx)
    return G, theta

def non_maximal_suppression(G, theta):
    M, N = G.shape
    suppressed = np.zeros((M, N))
    angle = np.rad2deg(theta) % 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            a = angle[i, j]
            q = r = 0
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif 22.5 <= a < 67.5:
                q, r = G[i+1, j-1], G[i-1, j+1]
            elif 67.5 <= a < 112.5:
                q, r = G[i+1, j], G[i-1, j]
            elif 112.5 <= a < 157.5:
                q, r = G[i-1, j-1], G[i+1, j+1]
            suppressed[i, j] = G[i, j] if G[i, j] >= q and G[i, j] >= r else 0
    return suppressed

def auto_threshold_percentile(G_suppressed, low_p=30, high_p=97):
    high = np.percentile(G_suppressed, high_p)
    low = np.percentile(G_suppressed, low_p)
    if low > high:
        low = high * 0.5
    low = max(low, 0.001)
    high = min(high, 0.8)
    return low, high

def hysteresis_thresholding(G, low_thresh, high_thresh):
    M, N = G.shape
    result = np.zeros((M, N), dtype=np.uint8)
    strong = G >= high_thresh
    weak = (G >= low_thresh) & (G < high_thresh)
    result[strong] = 255
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if weak[i, j] and np.any(strong[i-1:i+2, j-1:j+2]):
                result[i, j] = 255
    return result

def auto_select_parameters(img, kernel_options=[5, 7, 9, 11]):
    best_alpha, best_sigma, best_kernel, best_score = 0.5, 1.0, 5, -1
    for kernel_size in kernel_options:
        for alpha in [0.3, 0.5, 0.7]:
            for sigma in [0.5, 1.0, 2.0]:
                smoothed = apply_gaussian_filter(img, sigma)
                G, _ = compute_gradients(smoothed, alpha, kernel_size)
                mean_g = np.mean(G)
                std_g = np.std(G)
                score = std_g / (mean_g + 1e-5)
                if score > best_score:
                    best_alpha, best_sigma, best_kernel, best_score = alpha, sigma, kernel_size, score
    return best_alpha, best_sigma, best_kernel

class EdgeDetectionGUI:
    def __init__(self, master, image_path):
        self.master = master
        self.img = load_image(image_path)
        self.recommended_alpha, self.recommended_sigma, self.kernel_size = auto_select_parameters(self.img)

        Label(master, text=f"Recommended alpha: {self.recommended_alpha}").pack()
        self.alpha_slider = Scale(master, from_=0.1, to=1.0, resolution=0.1, orient=HORIZONTAL, label="alpha")
        self.alpha_slider.set(self.recommended_alpha)
        self.alpha_slider.pack()

        Label(master, text=f"Recommended sigma: {self.recommended_sigma}").pack()
        self.sigma_slider = Scale(master, from_=0.5, to=2.0, resolution=0.1, orient=HORIZONTAL, label="sigma")
        self.sigma_slider.set(self.recommended_sigma)
        self.sigma_slider.pack()

        Button(master, text="Detect Edges", command=self.detect_edges).pack()

    def detect_edges(self):
        alpha = self.alpha_slider.get()
        sigma = self.sigma_slider.get()

        img_filtered = apply_gaussian_filter(self.img, sigma)
        G, theta = compute_gradients(img_filtered, alpha, self.kernel_size)
        G_supp = non_maximal_suppression(G, theta)
        low, high = auto_threshold_percentile(G_supp)

        edges = hysteresis_thresholding(G_supp, low, high)
        Image.fromarray(edges).save("result_edges.png")

        edges_cv = cv2.Canny((img_filtered * 255).astype(np.uint8), 50, 150)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(self.img, cmap='gray')
        ax[0].set_title("Original")
        ax[1].imshow(edges, cmap='gray')
        ax[1].set_title(f"Fractional Edges (α={alpha}, σ={sigma})")
        ax[2].imshow(edges_cv, cmap='gray')
        ax[2].set_title("OpenCV Canny")
        for a in ax: a.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Оберіть зображення", filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
    if file_path:
        root = tk.Tk()
        root.title("Fractional Edge Detection (with sliders)")
        app = EdgeDetectionGUI(root, file_path)
        root.mainloop()
    else:
        print("Файл не вибрано.")
