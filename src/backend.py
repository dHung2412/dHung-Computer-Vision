from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import pickle
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": ["Content-Type"]}})

# Danh sách tên các biển báo (từ classification report)
TRAFFIC_SIGN_CLASSES = [
    'cấm xe ô tô tải vượt',           # Class 0
    'bắt đầu đường ưu tiên',          # Class 1
    'stop',                           # Class 2
    'đường cấm',                      # Class 3
    'cấm ô tô tải',                   # Class 4
    'cấm đi ngược chiều',             # Class 5
    'cảnh báo đường gập khúc',        # Class 6
    'đường có gò giảm sốc',           # Class 7
    'vòng xuyến'                      # Class 8
]

def load_model(model_path, model_name):
    """Load pre-trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"{model_name} Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        return None

# Load both models khi khởi động
rf_model = r"D:\Project\2025-2026\computer_vision\models\best_svm_model.pkl"
svm_model = r"D:\Project\2025-2026\computer_vision\models\best_rf_model.pkl"

def preprocess_image(image, target_size=(32, 32)):
    """Preprocess image for model prediction"""
    # Resize image
    resized = cv2.resize(image, target_size)
    # Convert to RGB if needed
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    # Normalize
    normalized = resized.astype('float32') / 255.0
    # Flatten
    flattened = normalized.flatten()
    return flattened

def detect_traffic_signs_cascade(image):
    """
    Detect traffic signs using improved color-based detection
    Returns list of bounding boxes (x, y, w, h)
    """
    bboxes = []
    height, width = image.shape[:2]
    
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detect red signs (stop signs, prohibitory signs) - wider range
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Detect blue signs (mandatory signs) - wider range
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Detect yellow signs (warning signs)
    lower_yellow = np.array([15, 70, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Detect white signs
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks
    combined_mask = red_mask + blue_mask + yellow_mask + white_mask
    
    # Morphological operations
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours")
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:  # Lower threshold to detect smaller signs
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (signs are usually circular or square)
            aspect_ratio = float(w) / h
            
            # Filter by size (not too small, not too large)
            size_ratio = (w * h) / (width * height)
            
            if 0.4 < aspect_ratio < 2.5 and size_ratio < 0.5:
                bboxes.append((x, y, w, h))
                print(f"Detected sign at ({x}, {y}, {w}, {h}), area={area}, aspect_ratio={aspect_ratio:.2f}")
    
    print(f"Total valid bounding boxes: {len(bboxes)}")
    return bboxes

def classify_sign(image, bbox, model):
    """Classify a detected sign region"""
    x, y, w, h = bbox
    sign_roi = image[y:y+h, x:x+w]
    
    # Preprocess ROI
    features = preprocess_image(sign_roi)
    features = features.reshape(1, -1)
    
    # Predict
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = np.max(probability)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in classification: {e}")
        return 0, 0.0

def draw_predictions(image, detections, color, model_name):
    """Draw bounding boxes and labels on image"""
    output_image = image.copy()
    
    for bbox, class_id, confidence in detections:
        x, y, w, h = bbox
        
        # Get class name
        class_name = TRAFFIC_SIGN_CLASSES[class_id] if class_id < len(TRAFFIC_SIGN_CLASSES) else f"Class {class_id}"
        
        # Draw bounding box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        
        # Prepare label: Tên biển báo + Tỉ lệ dự đoán
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text (above bounding box)
        bg_y1 = max(0, y - text_height - 10)
        bg_y2 = y
        cv2.rectangle(output_image, (x, bg_y1), (x + text_width + 10, bg_y2), color, -1)
        
        # Draw text in white color for better visibility
        cv2.putText(output_image, label, (x + 5, y - 8), font, font_scale, (255, 255, 255), thickness)
    
    return output_image

def process_detection(image, model, model_name, color):
    """Process detection with a specific model"""
    if model is None:
        return None, f'{model_name} model not loaded'
    
    try:
        print(f"\n{'='*50}")
        print(f"Processing with {model_name}")
        print(f"{'='*50}")
        
        # Detect traffic signs
        bboxes = detect_traffic_signs_cascade(image)
        
        if len(bboxes) == 0:
            print(f"WARNING: No traffic signs detected in image!")
            # If no detection, return original image with warning text
            result_image = image.copy()
            cv2.putText(result_image, "No traffic signs detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result_image, None
        
        # Classify each detection
        detections = []
        for i, bbox in enumerate(bboxes):
            class_id, confidence = classify_sign(image, bbox, model)
            print(f"Sign {i+1}: Class={class_id} ({TRAFFIC_SIGN_CLASSES[class_id] if class_id < len(TRAFFIC_SIGN_CLASSES) else 'Unknown'}), Confidence={confidence:.3f}")
            
            if confidence > 0.1:  # Lower threshold for testing
                detections.append((bbox, class_id, confidence))
            else:
                print(f"  -> Rejected: confidence too low")
        
        print(f"Total detections after filtering: {len(detections)}")
        
        # Draw predictions
        result_image = draw_predictions(image, detections, color, model_name)
        
        return result_image, None
    
    except Exception as e:
        print(f"ERROR in process_detection: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

@app.route('/detect/svm', methods=['POST'])
def detect_svm():
    """SVM detection endpoint"""
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    
    try:
        # Read image
        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'Invalid image'}, 400
        
        # Process with SVM
        result_image, error = process_detection(image, svm_model, "SVM", (0, 255, 0))
        
        if error:
            return {'error': error}, 500
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', result_image)
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        
        return send_file(io_buf, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error processing image with SVM: {e}")
        return {'error': str(e)}, 500

@app.route('/detect/rf', methods=['POST'])
def detect_rf():
    """Random Forest detection endpoint"""
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    
    try:
        # Read image
        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'Invalid image'}, 400
        
        # Process with Random Forest
        result_image, error = process_detection(image, rf_model, "Random Forest", (255, 0, 0))
        
        if error:
            return {'error': error}, 500
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', result_image)
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        
        return send_file(io_buf, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error processing image with Random Forest: {e}")
        return {'error': str(e)}, 500

@app.route('/detect/both', methods=['POST'])
def detect_both():
    """
    Detect using both models and return both results
    Returns JSON with base64 encoded images
    """
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    
    try:
        # Read image
        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'Invalid image'}, 400
        
        results = {}
        
        # Process with SVM
        svm_result, svm_error = process_detection(image.copy(), svm_model, "SVM", (0, 255, 0))
        if svm_error:
            results['svm'] = {'error': svm_error}
        else:
            _, buffer = cv2.imencode('.jpg', svm_result)
            import base64
            results['svm'] = {
                'image': base64.b64encode(buffer).decode('utf-8'),
                'success': True
            }
        
        # Process with Random Forest
        rf_result, rf_error = process_detection(image.copy(), rf_model, "Random Forest", (255, 0, 0))
        if rf_error:
            results['rf'] = {'error': rf_error}
        else:
            _, buffer = cv2.imencode('.jpg', rf_result)
            import base64
            results['rf'] = {
                'image': base64.b64encode(buffer).decode('utf-8'),
                'success': True
            }
        
        return results
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return {'error': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'models': {
            'svm': svm_model is not None,
            'random_forest': rf_model is not None
        }
    }

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Unified Traffic Sign Detection Backend")
    print("=" * 50)
    print(f"SVM Model loaded: {svm_model is not None}")
    print(f"Random Forest Model loaded: {rf_model is not None}")
    print("\nAvailable endpoints:")
    print("  - POST /detect/svm       : Detect using SVM")
    print("  - POST /detect/rf        : Detect using Random Forest")
    print("  - POST /detect/both      : Detect using both models")
    print("  - GET  /health           : Check server health")
    print("\nServer running on http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)