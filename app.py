import gradio as gr
from fastai.vision.all import *
import warnings

# --- 0. SETUP & MODEL LOADING ---

# Suppress the specific UserWarning from fastai about pickle
warnings.filterwarnings(
    "ignore",
    message="load_learner` uses Python's insecure pickle module.*",
    category=UserWarning,
)

# Load the trained model
learn = load_learner("export.pkl")

# Map the internal vocab names to the desired display names
LABEL_MAP = {"birds": "Bird", "planes": "Plane", "superman": "Superman"}

# --- 1. CORE PREDICTION FUNCTION ---


def predict(img):
    """
    Takes an image, gets predictions from the fastai model, and returns
    a dictionary with formatted labels and their corresponding probabilities.
    """
    # Get predictions from the model
    pred, pred_idx, probs = learn.predict(img)

    # Create a dictionary of probabilities with the original labels
    predictions = dict(zip(learn.dls.vocab, map(float, probs)))

    # Create a new dictionary with the formatted labels for display
    return {LABEL_MAP.get(k, k.capitalize()): v for k, v in predictions.items()}


# --- 2. DEFINE GRADIO CONTENT & LAYOUT ---

# Project description and details using Markdown for rich formatting
project_title = "# Is it a Bird? Is it a Plane? No, It's Superman!"
project_author = "Created by: **Von Mendres (馬盛中)**"

project_description = """
### **Project Description**
This project showcases a Computer Vision model that correctly distinguishes between images of birds, planes, and Superman. The model was created by fine-tuning a **ResNet34** architecture, achieving a **98% accuracy rate** on the validation set. The concept is inspired by the classic joke from the Superman comics, demonstrating a practical application of deep learning for image classification.

This application is built with Python and deployed as an interactive web app using Gradio and Hugging Face Spaces.
"""

technical_info = """
### **Technical Details**
* **Project Type:** Computer Vision / Image Classification
* **Model:** ResNet34 (fine-tuned)
* **Accuracy:** **98%** on the validation dataset
* **Key Technologies:**
    * **PyTorch:** The core deep learning framework.
    * **FastAI:** A high-level library used for streamlined model training and inference.
    * **Gradio:** The library used to build and deploy this interactive web interface.
    * **Hugging Face Spaces:** The platform hosting this live demonstration.
"""

footer = "July 2025 | Von Mendres (馬盛中)"

example_images = [
    "examples/example_bird.jpg",
    "examples/example_plane.jpg",
    "examples/example_superman.jpg",
]

# --- 3. BUILD THE GRADIO BLOCKS INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    # Main Title
    gr.Markdown(project_title)
    gr.Markdown(project_author)

    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload an Image", height=350)
            submit_btn = gr.Button("Classify Image", variant="primary")

            # Example images section
            gr.Examples(
                examples=example_images,
                inputs=input_image,
                examples_per_page=3,  # Makes images larger
                label="Click an example to try it out!",
            )

        # Output Column
        with gr.Column(scale=1):
            gr.Markdown("### **Prediction Results**")
            output_label = gr.Label(num_top_classes=3, label="Model Confidence")

    # Accordion for Project Details
    with gr.Accordion("About this Project & Technical Info", open=True):
        gr.Markdown(project_description)
        gr.Markdown(technical_info)

    # Footer
    gr.Markdown(footer)

    # --- 4. DEFINE COMPONENT INTERACTIONS ---
    submit_btn.click(fn=predict, inputs=input_image, outputs=output_label)

# --- 5. LAUNCH THE APP ---
if __name__ == "__main__":
    demo.launch()
