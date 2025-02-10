import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

example_images = [
    "image1.jpg",
    "image2.jpg",
]

def predict(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image)

    predictions = []
    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        conf = box.conf.item()
        predictions.append(f"Disease: {cls}, Confidence: {conf:.2f}")

    annotated_frame = results[0].plot()

    return annotated_frame, "\n".join(predictions)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Tomato Leaf Image"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Image"),
        gr.Textbox(label="Detection Details"),
    ],
    examples=example_images,
    title="Detection of Tomato Leaf Diseases by YOLOv5",
    description=(
        "Detect diseases by uploading a picture of a tomato leaf."
        "The model will predict the type of disease (if any) and annotate the image with bounding boxes."
    ),
)

interface.launch()
