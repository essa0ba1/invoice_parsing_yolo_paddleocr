# Invoice Parsing with YOLOv11 + PaddleOCR

An intelligent invoice parsing application that uses YOLOv11 for object detection and PaddleOCR for text extraction. This tool automatically detects and extracts key information from invoice images through a user-friendly web interface.

## Features

- **Object Detection**: Uses YOLOv11 to detect 16 different invoice fields including:
  - Client information (Name, Email, Phone)
  - Invoice details (Invoice Number, Invoice Date, Due Date)
  - Addresses (Billing Address, Shipping Address)
  - Financial information (Subtotal, Tax, Tax Percentage, Discount Percentage, Remise, Total)
  - Products list
  - Header information

- **Text Extraction**: Leverages PaddleOCR for accurate text recognition from detected regions
- **Web Interface**: Built with Gradio for easy-to-use image upload and visualization
- **Visual Annotations**: Displays bounding boxes on detected invoice fields

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file `best.pt` in the project directory (this is a trained YOLOv11 model)

## Usage

1. Run the application:
```bash
python app.py
```

2. The Gradio interface will launch in your browser (typically at `http://127.0.0.1:7860`)

3. Upload an invoice image using the interface

4. Click the "Detect" button to process the invoice

5. View the results:
   - **Annotated Image**: Shows the original invoice with bounding boxes around detected fields
   - **Extracted Text (JSON)**: Displays all extracted information in JSON format

## Project Structure

```
invoice_parsing/
├── app.py              # Main Gradio application
├── ocr.py              # OCR text extraction module
├── best.pt             # Trained YOLOv11 model
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How It Works

1. **Image Preprocessing**: The uploaded invoice image is resized to 640x640 pixels for YOLO detection
2. **Object Detection**: YOLOv11 model detects invoice fields and returns bounding box coordinates
3. **Coordinate Scaling**: Bounding boxes are scaled back to the original image dimensions
4. **Text Extraction**: Each detected region is cropped and processed through PaddleOCR
5. **Result Compilation**: Extracted text is organized by field type and returned as JSON

## Detected Fields

The application can detect and extract the following invoice fields:

- `Discount_Percentage`
- `Due_Date`
- `Email_Client`
- `Name_Client`
- `Products`
- `Remise`
- `Subtotal`
- `Tax`
- `Tax_Precentage`
- `Tel_Client`
- `billing address`
- `header`
- `invoice date`
- `invoice number`
- `shipping address`
- `total`

## Technical Details

- **Detection Model**: YOLOv11 (Ultralytics)
- **OCR Engine**: PaddleOCR v2.7.0.0
- **Web Framework**: Gradio 5.49.1
- **Image Processing**: PIL/Pillow
- **Device**: CPU (configurable in code)

## Notes

- The model runs on CPU by default. For better performance, you can modify the device settings in `app.py` if GPU is available
- The confidence threshold for detection is set to 0.4 (configurable in the `predict` function)
- Image size for detection is set to 640x640 pixels by default

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Author

[Add your name/contact information here]

