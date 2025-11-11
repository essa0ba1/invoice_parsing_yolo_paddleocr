import gradio as gr 
from torch import device
from ultralytics import YOLO
import os
from PIL import Image
from ocr import get_text_from_bbox


model = YOLO("best.pt")
model = model.to('cpu')


names = {0: 'Discount_Percentage', 1: 'Due_Date', 2: 'Email_Client', 3: 'Name_Client', 4: 'Products', 5: 'Remise', 6: 'Subtotal', 7: 'Tax', 8: 'Tax_Precentage', 9: 'Tel_Client', 10: 'billing address', 11: 'header', 12: 'invoice date', 13: 
 'invoice number', 14: 'shipping address', 15: 'total'}


def predict(image, img_size=640):
    image = image.convert("RGB")
    w, h = image.size
    org_image = image.copy()  # Keep original image for annotation
    
    # Calculate correct scale factors: how much the image was scaled DOWN
    scale_x = w / img_size  # e.g., if w=1600, img_size=640, scale_x = 0.4
    scale_y = h /img_size # e.g., if h=1131, img_size=640, scale_y = 0.566
    scale = (scale_x, scale_y)
    
    
    
    image = image.resize((img_size, img_size))  # Resize image to 640x640
    results = model.predict(image, conf=0.4,imgsz=img_size, device='cpu', verbose=False)
    annotated_image = results[0].plot()
    
    # Get bounding boxes and class indices
    bboxes = results[0].boxes.xyxy.numpy()
    cls_indices = results[0].boxes.cls.numpy()
    
    # Extract text from detected regions using original image
    extracted_text = get_text_from_bbox(org_image, bboxes, cls_indices, names,scale)
    
    # Get the annotated image
    return Image.fromarray(annotated_image), extracted_text

with gr.Blocks() as demo:
    gr.Markdown("# Invoice Detection with YOLOv11 + paddleOCR")
    gr.Markdown("Upload an invoice image to detect and annotate bounding boxes.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Invoice Image", type="pil")
            predict_button = gr.Button("Detect")
        with gr.Column():
            output_image = gr.Image(label="Annotated Image", type="pil")
    
    with gr.Row():
        json_output = gr.JSON(label="Extracted Text (JSON)", visible=True)
    
    predict_button.click(predict, inputs=image_input, outputs=[output_image, json_output])

demo.launch()