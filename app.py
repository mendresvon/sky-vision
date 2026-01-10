import gradio as gr
from fastai.vision.all import *
import warnings

# silence fastai pickle warning
warnings.filterwarnings(
    "ignore",
    message="load_learner` uses Python's insecure pickle module.*",
    category=UserWarning,
)

# load the trained model
learn = load_learner("export.pkl")

# map internal labels to display names
LABEL_MAP = {"birds": "Bird", "planes": "Plane", "superman": "Superman"}


def predict(img):
    """
    predicts class probabilities for an image and formats labels
    """
    pred, pred_idx, probs = learn.predict(img)

    # combine labels with prediction probabilities
    predictions = dict(zip(learn.dls.vocab, map(float, probs)))

    # format labels and capitalize for the ui
    return {LABEL_MAP.get(k, k.capitalize()): v for k, v in predictions.items()}


# project metadata and description content
project_title = "# Is it a Bird? Is it a Plane? No, It's Superman!"
project_author = "Created by: **Von Breznev Mendres (馬盛中)**"

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

footer = "July 2025 | Von Breznev Mendres (馬盛中)"

example_images = [
    "examples/example_bird.jpg",
    "examples/example_plane.jpg",
    "examples/example_superman.jpg",
]

with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    gr.Markdown(project_title)
    gr.Markdown(project_author)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload an Image", height=350)
            submit_btn = gr.Button("Classify Image", variant="primary")

            # use larger examples for better visibility
            gr.Examples(
                examples=example_images,
                inputs=input_image,
                examples_per_page=3,
                label="Click an example to try it out!",
            )

        with gr.Column(scale=1):
            gr.Markdown("### **Prediction Results**")
            output_label = gr.Label(num_top_classes=3, label="Model Confidence")

    with gr.Accordion("About this Project & Technical Info", open=True):
        gr.Markdown(project_description)
        gr.Markdown(technical_info)

    gr.Markdown(footer)

    # link the click event to the prediction function
    submit_btn.click(fn=predict, inputs=input_image, outputs=output_label)

if __name__ == "__main__":
    demo.launch()
