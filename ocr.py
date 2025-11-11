from paddleocr import PaddleOCR
from PIL import Image 
import numpy as np
import os

# Try to enable ONNX with proper error handling


ocr =  PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            show_log=False,
            use_angle_cls=True,
            use_gpu=False,
            cpu_threads=2,
            lang='en'
        )




def get_text_from_bbox(image, bboxes, cls_index, names, scale):
    parsing = {}
    
    # Ensure image is PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    
    
    for bbox, indx in zip(bboxes, cls_index): 
        x1, y1, x2, y2 = bbox[:4]
        
        
        
        # Scale coordinates back to original image size
        x1_scaled = x1 * scale[0]
        y1_scaled = y1 * scale[1]
        x2_scaled = x2 * scale[0]
        y2_scaled = y2 * scale[1]
        
        
        # Convert to integers and ensure bounds
        x1_crop = max(0, int(x1_scaled))
        y1_crop = max(0, int(y1_scaled))
        x2_crop = min(image.width, int(x2_scaled))
        y2_crop = min(image.height, int(y2_scaled))
        
        if x2_crop > x1_crop and y2_crop > y1_crop:  # Check if valid crop region
            try:
               
                # Use PIL crop method: crop((left, top, right, bottom))
                cropped_image = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
                
                # Convert to numpy for PaddleOCR
                cropped_array = np.array(cropped_image)
                ocr_result = ocr.ocr(cropped_array, cls=True)
                raw_texts = []
                
                if ocr_result:  # Check if OCR returned results
                    for line in ocr_result:
                        for word_info in line:
                            text = word_info[1][0]  # Get just the text
                            raw_texts.append(text)
                
                full_text = " ".join(raw_texts) if raw_texts else "No text detected"
                if names[int(indx)] in parsing:
                   
                    parsing[names[int(indx)]] +=  '\n'+full_text
                else:       
                    parsing[names[int(indx)]] = full_text                
            except Exception as e:
                print(f"  Error cropping: {e}")
                parsing[names[int(indx)]] = f"Error: {str(e)}"
        else:
            print(f"  Invalid crop region: x1={x1_crop}, y1={y1_crop}, x2={x2_crop}, y2={y2_crop}")
            parsing[names[int(indx)]] = "Invalid region"
    
    return parsing


