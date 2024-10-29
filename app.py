import cv2
import gradio as gr
import numpy as np

def detect_blur_and_tilt(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect if the image is blurry
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < 85  # Threshold for blur detection

    # Detect if the person has a tilted head
    # Assuming a simple approach based on edge detection and contours for the face tilt
    # Convert to grayscale and use Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Placeholder tilt detection: set to True if any major contour found
    head_tilted = len(contours) > 0

    return {
        "Is the image blurry?": is_blurry,
        "Is the head tilted?": head_tilted
    }

# Create Gradio Interface
iface = gr.Interface(
    fn=detect_blur_and_tilt,
    inputs="image",
    outputs="label",
    title="Blur & Head Tilt Detection"
)

if __name__ == "__main__":
    iface.launch()