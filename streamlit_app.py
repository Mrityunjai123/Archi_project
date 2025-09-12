import os
import streamlit as st
import cv2
import numpy as np
import fitz 
import io
from PIL import Image, ImageFile
import os
from pathlib import Path
import tempfile
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import json
import zipfile
import base64
import requests
import re
from typing import List, Dict, Any, Optional
import PyPDF2
from datetime import datetime
import gc  # For garbage collection

# Fix for large images and memory optimization
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuration
GEMINI_KEY = "AIzaSyARUWP7nhktvAnKqS3QgjEVEUKfSl_8iPw"  # Replace with your actual API key
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_KEY}"

# Cloud deployment optimizations
CLOUD_MODE = True
PDF_DPI = 200 if CLOUD_MODE else 600  # Reduced DPI for cloud
MAX_IMAGE_DIMENSION = 2000 if CLOUD_MODE else 8000  # Reduced for cloud
MAX_FILE_SIZE_MB = 25  # Maximum PDF file size in MB

# Comprehensive Livingston Township Zoning Requirements from PDF
LIVINGSTON_ZONING_REQUIREMENTS = {
    'R-1': {
        'use': 'Single Family',
        'front_setback': 75,  # feet
        'side_setback': 15,   # feet
        'rear_setback': 75,   # feet
        'height': 35,         # feet
        'hf_area': 6200,      # sq ft
        'hf_ratio': 15,       # percent
        'standard_lot': '150\' x 235\'',
        'min_lot_width': 150,  # feet
        'min_lot_area': 35250, # sq ft
        'min_floor_area': None
    },
    'R-2': {
        'use': 'Single Family',
        'front_setback': 60,
        'side_setback': 15,
        'rear_setback': 50,
        'height': 35,
        'hf_area': 4870,
        'hf_ratio': 18,
        'standard_lot': '125\' x 200\'',
        'min_lot_width': 125,
        'min_lot_area': 25000,
        'min_floor_area': None
    },
    'R-3': {
        'use': 'Single Family',
        'front_setback': 50,
        'side_setback': 10,
        'rear_setback': 40,
        'height': 35,
        'hf_area': 3520,
        'hf_ratio': 21,
        'standard_lot': '100\' x 150\'',
        'min_lot_width': 100,
        'min_lot_area': 15000,
        'min_floor_area': None
    },
    'R-4': {
        'use': 'Single Family',
        'front_setback': 40,
        'side_setback': 10,
        'rear_setback': 35,
        'height': 35,
        'hf_area': 3220,
        'hf_ratio': 30,
        'standard_lot': '75\' x 125\'',
        'min_lot_width': 75,
        'min_lot_area': 9375,
        'min_floor_area': None
    },
    'B': {
        'use': 'Central Business',
        'front_setback': None,
        'side_setback': None,
        'rear_setback': None,
        'height': 28,
        'hf_area': None,
        'hf_ratio': None,
        'standard_lot': None,
        'min_lot_width': None,
        'min_lot_area': None,
        'min_floor_area': None
    },
    'B-1': {
        'use': 'General Business',
        'front_setback': None,
        'side_setback': None,
        'rear_setback': None,
        'height': 28,
        'hf_area': None,
        'hf_ratio': None,
        'standard_lot': None,
        'min_lot_width': None,
        'min_lot_area': None,
        'min_floor_area': None
    },
    'B-2': {
        'use': 'Highway Business',
        'front_setback': 125,
        'side_setback': 12,
        'rear_setback': 100,
        'height': 28,
        'hf_area': None,
        'hf_ratio': None,
        'standard_lot': None,
        'min_lot_width': None,
        'min_lot_area': 5000,
        'min_floor_area': None
    },
    'P-B': {
        'use': 'Professional Building',
        'front_setback': 75,
        'side_setback': 25,
        'rear_setback': 100,
        'height': 28,
        'hf_area': None,
        'hf_ratio': None,
        'standard_lot': None,
        'min_lot_width': None,
        'min_lot_area': None,
        'min_floor_area': None
    },
    'P-B1': {
        'use': 'Professional Office District',
        'front_setback': 80,
        'side_setback': 100,
        'rear_setback': 100,
        'height': 40,
        'hf_area': None,
        'hf_ratio': None,
        'standard_lot': None,
        'min_lot_width': None,
        'min_lot_area': None,
        'min_floor_area': None
    },
    'I': {
        'use': 'Limited Industry',
        'front_setback': 50,
        'side_setback': 40,
        'rear_setback': 60,
        'height': 28,
        'hf_area': None,
        'hf_ratio': None,
        'standard_lot': None,
        'min_lot_width': None,
        'min_lot_area': 40000,
        'min_floor_area': None
    }
}

class PDFtoImageConverter:
    """Convert PDF to high-resolution images using PyMuPDF with cloud optimizations"""
    
    def __init__(self, dpi=PDF_DPI):
        self.dpi = dpi
    
    def convert_pdf_to_images(self, pdf_file, max_dimension=MAX_IMAGE_DIMENSION):
        try:
            # Use PyMuPDF instead of pdf2image
            if isinstance(pdf_file, bytes):
                pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
            else:
                pdf_document = fitz.open(pdf_file)
            
            processed_images = []
            
            # Limit number of pages for cloud deployment
            max_pages = 5 if CLOUD_MODE else len(pdf_document)
            actual_pages = min(len(pdf_document), max_pages)
            
            if len(pdf_document) > max_pages:
                st.warning(f"Processing first {max_pages} pages only (cloud limit)")
            
            for page_num in range(actual_pages):
                page = pdf_document.load_page(page_num)
                
                # Convert to image with specified DPI
                mat = fitz.Matrix(self.dpi/72, self.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                original_w, original_h = image.size
                
                st.info(f"Page {page_num+1}: Original size {original_w}x{original_h} ({(original_w*original_h)/1_000_000:.1f}M pixels)")
                
                # Resize if too large (more aggressive for cloud)
                if original_w > max_dimension or original_h > max_dimension:
                    if original_h > original_w:
                        new_h = max_dimension
                        new_w = int(original_w * (new_h / original_h))
                    else:
                        new_w = max_dimension
                        new_h = int(original_h * (new_w / original_w))
                    
                    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    st.info(f"Resized to: {new_w}x{new_h}")
                
                processed_images.append(image)
                
                # Clean up PyMuPDF objects
                pix = None
                
                # Force garbage collection for cloud deployment
                if CLOUD_MODE:
                    gc.collect()
            
            pdf_document.close()
            return processed_images
            
        except Exception as e:
            st.error(f"Error converting PDF: {str(e)}")
            if CLOUD_MODE:
                st.info("Try uploading a smaller PDF or reducing the number of pages")
            return []

class YOLOv8Pipeline:
    """YOLOv8 Inference Pipeline with Zone Detection and cloud optimizations"""
    
    def __init__(self, model_path=None):
        # Multiple model path options for different deployments
        possible_paths = [
            "model/new_yolov8_final.pt",     # GitHub structure
            "models/new_yolov8_final.pt",    # Alternative structure
            "custom_yolov8_final.pt",           # Root directory
        ] if model_path is None else [model_path]
        
        self.model = None
        self.custom_model = False
        
        # Try to load custom model
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    st.info(f"Loading custom model: {path}")
                    self.model = YOLO(path)
                    self.custom_model = True
                    self.classes = list(self.model.names.values()) if hasattr(self.model, 'names') else ["Architecture Layout", "Tables", "zone area"]
                    st.success(f"Custom model loaded successfully: {os.path.basename(path)}")
                    break
                except Exception as e:
                    st.warning(f"Could not load custom model from {path}: {e}")
                    continue
        
        # Fallback to default model
        if self.model is None:
            st.info("Using default YOLOv8 model")
            self.model = YOLO('yolov8n.pt')
            self.custom_model = False
            self.classes = list(self.model.names.values())
        
        self.colors = self._generate_colors(len(self.classes))
    
    def detect_zone_from_results(self, results):
        """Detect zoning designation from YOLO results"""
        detected_zones = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id < len(self.classes):
                    class_name = self.classes[class_id].upper()
                    
                    # Check if class name contains zone indicators
                    zone_patterns = ['R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'R-6', 
                                   'B-1', 'B-2', 'B', 'P-B', 'P-B1', 'P-B2', 'P-B3',
                                   'I', 'CI', 'AH', 'MU-1', 'MU-2', 'ZONE', 'ZONING']
                    
                    for pattern in zone_patterns:
                        if pattern in class_name:
                            detected_zones.append({
                                'zone': pattern,
                                'confidence': confidence,
                                'class_name': class_name,
                                'bbox': box.xyxy[0].cpu().numpy().tolist()
                            })
                            break
                    
                    # Also check if the class is specifically "zone area" or similar
                    if any(zone_word in class_name.lower() for zone_word in ['zone', 'zoning']):
                        detected_zones.append({
                            'zone': 'ZONE_AREA_DETECTED',
                            'confidence': confidence,
                            'class_name': class_name,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        })
        
        return detected_zones
    
    def _generate_colors(self, num_classes):
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = plt.cm.hsv(hue)[:3]
            bgr = tuple(int(255 * c) for c in rgb[::-1])
            colors.append(bgr)
        return colors
    
    def predict_image(self, image, conf_threshold=0.25):
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Cloud optimization: use smaller image for inference if too large
        if CLOUD_MODE and img_array.size > 2000*2000*3:
            h, w = img_array.shape[:2]
            scale_factor = min(1500/w, 1500/h)
            if scale_factor < 1:
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img_array = cv2.resize(img_array, (new_w, new_h))
                st.info(f"Reduced image size for YOLO inference: {new_w}x{new_h}")
        
        results = self.model(img_array, conf=conf_threshold)
        annotated_image = self.draw_predictions(img_array.copy(), results[0])
        
        return annotated_image, results[0]
    
    def draw_predictions(self, image, results):
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id < len(self.classes):
                    class_name = self.classes[class_id]
                else:
                    class_name = f"Class_{class_id}"
                
                color = self.colors[class_id % len(self.colors)]
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return image
    
    def extract_detected_regions(self, image, results, output_prefix="detected"):
        extracted_data = {
            'cropped_images': {},
            'detection_data': [],
            'summary': {
                'total_extractions': 0,
                'regions_by_class': {}
            }
        }
        
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        if results.boxes is not None and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id < len(self.classes):
                    class_name = self.classes[class_id]
                else:
                    class_name = f"Class_{class_id}"
                
                cropped_region = img_array[y1:y2, x1:x2]
                
                if cropped_region.size > 0:
                    cropped_pil = Image.fromarray(cropped_region)
                    filename = f"{output_prefix}_{class_name}_{i+1}_{confidence:.2f}"
                    
                    extracted_data['cropped_images'][filename] = cropped_pil
                    
                    detection_info = {
                        'filename': filename,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    
                    extracted_data['detection_data'].append(detection_info)
                    extracted_data['summary']['total_extractions'] += 1
                    extracted_data['summary']['regions_by_class'][class_name] = \
                        extracted_data['summary']['regions_by_class'].get(class_name, 0) + 1
        
        return extracted_data
    
    def get_detection_summary(self, results):
        summary = {
            'total_detections': 0,
            'detections_per_class': {},
            'detection_details': []
        }
        
        if results.boxes is not None and len(results.boxes) > 0:
            summary['total_detections'] = len(results.boxes)
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                if class_id < len(self.classes):
                    class_name = self.classes[class_id]
                else:
                    class_name = f"Class_{class_id}"
                
                summary['detections_per_class'][class_name] = summary['detections_per_class'].get(class_name, 0) + 1
                
                summary['detection_details'].append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return summary

class AnalysisValidator:
    """Reinforcement layer to ensure complete analysis execution"""
    
    def __init__(self):
        self.required_steps = {
            'pdf_processed': False,
            'yolo_completed': False,
            'regions_extracted': False,
            'architectures_identified': False,
            'all_analyses_completed': False,
            'results_generated': False
        }
        # Reduced requirements for cloud deployment
        self.analysis_requirements = {
            'min_lot_measurements': 2,  # Reduced from 4
            'min_setback_measurements': 1,
            'min_building_measurements': 1,
            'required_architecture_count': 1  # Reduced from 2
        }
    
    def validate_step(self, step_name: str, data: Any) -> Dict[str, Any]:
        """Validate completion of each step with detailed feedback"""
        validation_result = {
            'passed': False,
            'issues': [],
            'recommendations': [],
            'retry_needed': False
        }
        
        if step_name == 'pdf_processed':
            if data and len(data) > 0:
                validation_result['passed'] = True
                self.required_steps['pdf_processed'] = True
            else:
                validation_result['issues'].append("PDF processing failed or no pages extracted")
                validation_result['recommendations'].append("Retry PDF conversion with different settings")
        
        elif step_name == 'yolo_completed':
            if data and hasattr(data, 'boxes') and data.boxes is not None:
                detection_count = len(data.boxes)
                if detection_count >= 1:  # Reduced threshold for cloud
                    validation_result['passed'] = True
                    self.required_steps['yolo_completed'] = True
                else:
                    validation_result['issues'].append(f"Only {detection_count} detections found")
                    validation_result['recommendations'].append("Lower confidence threshold or check image quality")
            else:
                validation_result['issues'].append("YOLO detection failed or no detections found")
                validation_result['retry_needed'] = True
        
        elif step_name == 'regions_extracted':
            if data and data.get('cropped_images') and len(data['cropped_images']) >= 1:
                validation_result['passed'] = True
                self.required_steps['regions_extracted'] = True
                self.required_steps['architectures_identified'] = True
            else:
                validation_result['issues'].append("Region extraction failed or insufficient regions")
                validation_result['retry_needed'] = True
        
        elif step_name == 'measurements_analysis':
            issues_found = []
            
            # Validate lot measurements
            lot_count = len(data.get('lot_measurements', []))
            if lot_count < self.analysis_requirements['min_lot_measurements']:
                issues_found.append(f"Lot: {lot_count}/{self.analysis_requirements['min_lot_measurements']} measurements")
            
            # Validate setback measurements  
            setback_count = len(data.get('setback_measurements', []))
            if setback_count < self.analysis_requirements['min_setback_measurements']:
                issues_found.append(f"Setback: {setback_count}/{self.analysis_requirements['min_setback_measurements']} measurements")
            
            # Validate building measurements
            building_count = len(data.get('building_measurements', []))
            if building_count < self.analysis_requirements['min_building_measurements']:
                issues_found.append(f"Building: {building_count}/{self.analysis_requirements['min_building_measurements']} measurements")
            
            if issues_found:
                validation_result['issues'] = issues_found
                validation_result['recommendations'].append("Review Gemini AI prompts and image quality")
                validation_result['retry_needed'] = True
            else:
                validation_result['passed'] = True
        
        return validation_result

class EnhancedLotMeasurementsDetector:
    """Enhanced detector for all lot polygon dimensions with label recognition"""
    
    def detect_lot_measurements(self, image_path_or_pil) -> Dict[str, Any]:
        try:
            if isinstance(image_path_or_pil, (str, Path)):
                with open(image_path_or_pil, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                # PIL Image - optimize for cloud
                img_buffer = io.BytesIO()
                # Reduce quality for cloud deployment
                if CLOUD_MODE:
                    # Resize if too large
                    w, h = image_path_or_pil.size
                    if w > 1500 or h > 1500:
                        scale = min(1500/w, 1500/h)
                        new_size = (int(w*scale), int(h*scale))
                        image_path_or_pil = image_path_or_pil.resize(new_size, Image.Resampling.LANCZOS)
                
                image_path_or_pil.save(img_buffer, format='PNG', optimize=True)
                img_buffer.seek(0)
                base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Failed to encode image: {e}'}
        
        # In EnhancedLotMeasurementsDetector.detect_lot_measurements()
        enhanced_lot_prompt = """
CRITICAL LOT BOUNDARY MEASUREMENTS DETECTION

IMPORTANT: ONLY detect measurements that define the PROPERTY BOUNDARIES, NOT building dimensions.

WHAT TO FIND:
1. LOT AREA: "AREA = X SF", "X SQUARE FEET", "X ACRES"
2. LOT BOUNDARY DIMENSIONS: Property perimeter measurements
   - Street frontage dimensions
   - Side property line dimensions  
   - Rear property line dimensions
   - Curved boundary measurements (radius, arc length)

WHAT TO IGNORE:
- Building dimensions (these are NOT lot measurements)
- Interior building measurements
- Setback distances
- Any dimension that measures building parts

BOUNDARY IDENTIFICATION:
- Look for measurements along the OUTER PERIMETER of the entire property
- Property boundaries are usually shown as thick lines around the entire lot
- May include bearings like "S 28°-02'-00" W" with distances
- Street boundaries, side boundaries, rear boundaries

CRITICAL: If a measurement is between 10-100 feet and appears to be measuring a building or structure, REJECT it as it's likely a building dimension, not a lot boundary.

Return JSON with ONLY lot boundary measurements:
[
  {
    "text": "S 28°-02'-00" W 505.0'",
    "value": 505.0,
    "unit": "FT",
    "type": "lot_boundary",
    "method": "surveyor_bearing"
  },
  {
    "text": "AREA=94,195 S.F.",
    "value": 94195,
    "unit": "SF", 
    "type": "lot_area",
    "method": "explicit_label"
  }
]

REJECT building dimensions that are typically 15-50 feet - these are NOT lot measurements.
"""
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": enhanced_lot_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }]
        }
        
        try:
            # Reduced timeout for cloud deployment
            timeout = 60 if CLOUD_MODE else 90
            response = requests.post(GEMINI_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    measurements = self._extract_lot_measurements(content)
                    measurements = self._calculate_missing_area(measurements)
                    
                    return {
                        'success': True,
                        'measurements': measurements,
                        'raw_response': content
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_lot_measurements(self, response: str) -> List[Dict]:
        measurements = []
    
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                raw_data = json.loads(json_content)
            
                for item in raw_data:
                    # Add None checks for all values
                    text_val = item.get('text', '')
                    value_val = item.get('value')
                    unit_val = item.get('unit', 'FT')
                    type_val = item.get('type', 'unknown')
                    method_val = item.get('method', 'not_specified')
                
                    # Skip if value is None or not convertible to float
                    if value_val is None:
                        continue
                
                    try:
                        value_float = float(value_val)
                        if value_float <= 0:  # Skip zero or negative values
                            continue
                    except (ValueError, TypeError):
                        continue
                
                    measurement = {
                        'text': str(text_val) if text_val is not None else '',
                        'value': value_float,
                        'unit': str(unit_val) if unit_val is not None else 'FT',
                        'type': str(type_val) if type_val is not None else 'unknown',
                        'method': str(method_val) if method_val is not None else 'not_specified'
                    }
                    measurements.append(measurement)
            
                if measurements:
                    return measurements
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log the error for debugging
            st.warning(f"JSON parsing failed for lot measurements: {str(e)}")
    
        return self._enhanced_fallback_extraction(response)
    
    def _enhanced_fallback_extraction(self, text: str) -> List[Dict]:
        measurements = []
        
        labeled_patterns = [
            (r'L\s*=\s*(\d{1,3}\.?\d{0,2})[\'\"]*', 'FT', 'lot_width', 'labeled_dimension'),
            (r'B\s*=\s*(\d{1,3}\.?\d{0,2})[\'\"]*', 'FT', 'lot_depth', 'labeled_dimension'),
            (r'W\s*=\s*(\d{1,3}\.?\d{0,2})[\'\"]*', 'FT', 'lot_width', 'labeled_dimension'),
            (r'D\s*=\s*(\d{1,3}\.?\d{0,2})[\'\"]*', 'FT', 'lot_depth', 'labeled_dimension'),
            (r'R\s*=\s*(\d{1,3}\.?\d{0,2})[\'\"]*', 'FT', 'lot_curved_boundary', 'radius_measurement')
        ]
        
        area_patterns = [
            (r'AREA\s*=\s*(\d{1,5}[,\s]*\d*)\s*S\.?F\.?', 'SF', 'lot_area', 'explicit_label'),
            (r'(\d*\.?\d+)\s*ACRES?', 'ACRES', 'lot_area', 'explicit_label'),
            (r'(\d{1,5}[,\s]*\d*)\s*(?:SQ\.?\s*FT\.?|SQUARE\s+FEET)', 'SF', 'lot_area', 'explicit_label')
        ]
        
        dimension_patterns = [
            (r'(\d{1,3}\.?\d{0,2})\s*FT\.?\s*(?:FRONTAGE|FRONT)', 'FT', 'lot_frontage', 'frontage_measurement'),
            (r'(\d{1,3}\.?\d{0,2})\s*FT\.?\s*(?:DEPTH|DEEP)', 'FT', 'lot_depth', 'depth_measurement'),
            (r'(\d{1,3}\.?\d{0,2})\s*FT\.?', 'FT', 'lot_dimension', 'general_dimension'),
            (r"(\d{1,3}\.?\d{0,2})'", 'FT', 'lot_dimension', 'general_dimension')
        ]
        
        all_patterns = labeled_patterns + area_patterns + dimension_patterns
        
        for pattern, unit, default_type, method in all_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '').replace(' ', '')
                    value = float(value_str)
                    
                    if unit == 'ACRES':
                        value *= 43560
                        unit = 'SF'
                        default_type = 'lot_area'
                    
                    if default_type == 'lot_dimension':
                        if 40 <= value <= 200:
                            default_type = 'lot_width'
                        elif 60 <= value <= 400:
                            default_type = 'lot_depth'
                        elif value > 250:
                            default_type = 'lot_curved_boundary'
                        else:
                            default_type = 'lot_side_dimension'
                    
                    measurement = {
                        'text': match.group(0).strip(),
                        'value': value,
                        'unit': unit,
                        'type': default_type,
                        'method': method
                    }
                    measurements.append(measurement)
                    
                except (ValueError, IndexError):
                    continue
        
        # Remove duplicates
        unique_measurements = []
        for measure in measurements:
            is_duplicate = False
            for existing in unique_measurements:
                if (abs(existing['value'] - measure['value']) < 0.5 and 
                    existing['unit'] == measure['unit']):
                    if measure['method'] == 'labeled_dimension' and existing['method'] != 'labeled_dimension':
                        unique_measurements.remove(existing)
                        unique_measurements.append(measure)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_measurements.append(measure)
        
        return unique_measurements
    
    def _calculate_missing_area(self, measurements: List[Dict]) -> List[Dict]:
        has_explicit_area = any(m['type'] == 'lot_area' for m in measurements)
        if has_explicit_area:
            return measurements
        
        widths = [m for m in measurements if m['type'] in ['lot_width', 'lot_frontage']]
        depths = [m for m in measurements if m['type'] == 'lot_depth']
        
        if widths and depths:
            min_width = min(widths, key=lambda x: x['value'])
            min_depth = min(depths, key=lambda x: x['value'])
            
            calculated_area = min_width['value'] * min_depth['value']
            
            calculated_measurement = {
                'text': f"Calculated: {min_width['value']:.2f}' × {min_depth['value']:.2f}'",
                'value': calculated_area,
                'unit': 'SF',
                'type': 'lot_area_calculated',
                'method': 'width_x_depth_minimum'
            }
            
            measurements.append(calculated_measurement)
        
        return measurements

class EnhancedSetbackMeasurementsDetector:
    """Enhanced detector for accurate dwelling-to-boundary setback measurements"""
    
    def detect_setback_measurements(self, image_path_or_pil) -> Dict[str, Any]:
        try:
            if isinstance(image_path_or_pil, (str, Path)):
                with open(image_path_or_pil, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                img_buffer = io.BytesIO()
                # Cloud optimization
                if CLOUD_MODE:
                    w, h = image_path_or_pil.size
                    if w > 1500 or h > 1500:
                        scale = min(1500/w, 1500/h)
                        new_size = (int(w*scale), int(h*scale))
                        image_path_or_pil = image_path_or_pil.resize(new_size, Image.Resampling.LANCZOS)
                
                image_path_or_pil.save(img_buffer, format='PNG', optimize=True)
                img_buffer.seek(0)
                base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Failed to encode image: {e}'}
        
        accurate_setback_prompt = """
CRITICAL: USE "DRIVEWAY" TEXT TO DETERMINE FRONT ORIENTATION, THEN FIND ALL SETBACKS

STEP 1: FIND "DRIVEWAY" TEXT IN THE IMAGE
• Look for the word "DRIVEWAY" or "DRIVE" in text labels
• The side of the lot with the driveway connection is the FRONT of the house
• This establishes the front orientation definitively

STEP 2: IDENTIFY THE DWELLING BUILDING
• Find the building footprint (hatched/shaded polygon)
• Main residential structure (not garage, unless attached to main house)

STEP 3: ESTABLISH ORIENTATION BASED ON DRIVEWAY
• FRONT: The side where driveway connects to the dwelling
• REAR: Opposite side from the driveway connection
• LEFT SIDE: Left side when facing the house from the driveway
• RIGHT SIDE: Right side when facing the house from the driveway

STEP 4: FIND SETBACK MEASUREMENTS (DWELLING TO BOUNDARY)
Look for dimension lines that show distance from:
• Dwelling front wall to front property boundary (where driveway enters lot)
• Dwelling rear wall to rear property boundary (opposite from driveway)
• Dwelling left wall to left property boundary
• Dwelling right wall to right property boundary

SETBACK MEASUREMENT INDICATORS:
• Dimension lines with arrows from building wall to property boundary
• Gap measurements between dwelling outline and lot perimeter lines
• Any measurement showing clearance from building to property line

STRICT REQUIREMENTS:
• Must be measurement from building structure to lot boundary
• Must use DRIVEWAY location to determine front/rear orientation
• Reject lot boundary dimensions or building interior dimensions

Return JSON with setbacks based on driveway orientation:
[
  {
    "text": "X.X' setback from dwelling to front boundary",
    "value": X.X,
    "unit": "FT",
    "type": "front_setback",
    "orientation_method": "driveway_indicates_front",
    "confidence": "high"
  },
  {
    "text": "Y.Y' setback from dwelling to rear boundary", 
    "value": Y.Y,
    "unit": "FT",
    "type": "rear_setback",
    "orientation_method": "opposite_from_driveway",
    "confidence": "high"
  }
]

MANDATORY: Use DRIVEWAY text location as the definitive front orientation indicator.
"""
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": accurate_setback_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }]
        }
        
        try:
            timeout = 60 if CLOUD_MODE else 90
            response = requests.post(GEMINI_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    measurements = self._extract_verified_setback_measurements(content)
                    return {
                        'success': True,
                        'measurements': measurements,
                        'raw_response': content
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_verified_setback_measurements(self, response: str) -> List[Dict]:
        """Extract only verified dwelling-to-boundary setback measurements"""
        measurements = []
    
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                raw_data = json.loads(json_content)
            
                for item in raw_data:
                    # Add None checks for all values
                    text_val = item.get('text', '')
                    value_val = item.get('value')
                    unit_val = item.get('unit', 'FT')
                    type_val = item.get('type', 'unknown')
                    measurement_type_val = item.get('measurement_type', 'unspecified')
                    confidence_val = item.get('confidence', 'medium')
                
                    # Skip if value is None or not convertible to float
                    if value_val is None:
                        continue
                
                    try:
                        value_float = float(value_val)
                        if value_float <= 0:  # Skip zero or negative values
                            continue
                    except (ValueError, TypeError):
                        continue
                
                    # Create item dict for validation
                    validation_item = {
                        'value': value_float,
                        'measurement_type': str(measurement_type_val) if measurement_type_val is not None else '',
                        'type': str(type_val) if type_val is not None else 'unknown'
                    }
                
                    # Strict validation for setbacks
                    if self._validate_setback_measurement(validation_item):
                        measurement = {
                            'text': str(text_val) if text_val is not None else '',
                            'value': value_float,
                            'unit': str(unit_val) if unit_val is not None else 'FT',
                            'type': str(type_val) if type_val is not None else 'unknown',
                            'measurement_type': str(measurement_type_val) if measurement_type_val is not None else 'unspecified',
                            'confidence': str(confidence_val) if confidence_val is not None else 'medium'
                        }
                        measurements.append(measurement)
            
                if measurements:
                    return measurements
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log the error for debugging
            st.warning(f"JSON parsing failed for setback measurements: {str(e)}")
    
        return self._fallback_setback_extraction(response)
    
    def _validate_setback_measurement(self, item: Dict) -> bool:
        """Strict validation for setback measurements"""
        try:
            value = float(item.get('value', 0))
            measurement_type = item.get('measurement_type', '')
            
            # Must be in reasonable setback range
            if not (3 <= value <= 100):
                return False
            
            # Must be classified as dwelling-to-boundary measurement
            required_keywords = ['dwelling', 'boundary', 'setback']
            measurement_text = (measurement_type + ' ' + item.get('type', '')).lower()
            
            if not any(keyword in measurement_text for keyword in required_keywords):
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _fallback_setback_extraction(self, text: str) -> List[Dict]:
        """Fallback extraction with strict dwelling-to-boundary validation"""
        measurements = []
        
        # Only look for explicitly labeled setbacks
        setback_patterns = [
            (r'(\d{1,2}\.?\d{0,2})[\'\"]*\s*(?:FRONT\s+)?SETBACK', 'FT', 'front_setback'),
            (r'(\d{1,2}\.?\d{0,2})[\'\"]*\s*(?:REAR\s+)?SETBACK', 'FT', 'rear_setback'),
            (r'(\d{1,2}\.?\d{0,2})[\'\"]*\s*(?:SIDE\s+)?SETBACK', 'FT', 'side_setback'),
            (r'SETBACK\s*[=:]?\s*(\d{1,2}\.?\d{0,2})[\'\"]*', 'FT', 'setback'),
            (r'(\d{1,2}\.?\d{0,2})[\'\"]*\s*TO\s+PROPERTY\s+LINE', 'FT', 'property_line_setback'),
            (r'(\d{1,2}\.?\d{0,2})[\'\"]*\s*FROM\s+BUILDING', 'FT', 'building_setback')
        ]
        
        for pattern, unit, setback_type in setback_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    
                    # Strict validation
                    if not (3 <= value <= 100):
                        continue
                    
                    measurement = {
                        'text': match.group(0).strip(),
                        'value': value,
                        'unit': unit,
                        'type': setback_type,
                        'measurement_type': 'dwelling_to_boundary_verified',
                        'confidence': 'medium'
                    }
                    measurements.append(measurement)
                    
                except (ValueError, IndexError):
                    continue
        
        return measurements

class EnhancedBuildingMeasurementsDetector:
    """Enhanced detector for all building measurements with comprehensive polygon analysis"""
    
    def detect_building_measurements(self, image_path_or_pil) -> Dict[str, Any]:
        try:
            if isinstance(image_path_or_pil, (str, Path)):
                with open(image_path_or_pil, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                img_buffer = io.BytesIO()
                # Cloud optimization
                if CLOUD_MODE:
                    w, h = image_path_or_pil.size
                    if w > 1500 or h > 1500:
                        scale = min(1500/w, 1500/h)
                        new_size = (int(w*scale), int(h*scale))
                        image_path_or_pil = image_path_or_pil.resize(new_size, Image.Resampling.LANCZOS)
                
                image_path_or_pil.save(img_buffer, format='PNG', optimize=True)
                img_buffer.seek(0)
                base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Failed to encode image: {e}'}
        
        # In EnhancedBuildingMeasurementsDetector.detect_building_measurements()
        enhanced_building_prompt = """
BUILDING/DWELLING MEASUREMENTS DETECTION

CRITICAL: Find measurements that define the BUILDING STRUCTURE only.

STEP 1: IDENTIFY THE DWELLING
- Look for building footprint (hatched/shaded area)
- Labels: "DWELLING", "HOUSE", "RESIDENCE"
- Distinguished from "GARAGE", "SHED", "PORCH"

STEP 2: FIND BUILDING MEASUREMENTS
1. EXPLICIT BUILDING AREA:
   • "BUILDING AREA = X SF"
   • "FLOOR AREA = X SF"
   • "FOOTPRINT = X SF"

2. BUILDING DIMENSIONS:
   • Exterior building width/length
   • Overall building footprint dimensions
   • Building height or stories

3. AVOID:
   • Lot boundary measurements
   • Setback measurements
   • Interior room dimensions

RETURN ONLY UNIQUE MEASUREMENTS:
[
  {
    "text": "Building Area: 1,200 SF",
    "value": 1200,
    "unit": "SF",
    "type": "dwelling_area",
    "component": "main_structure"
  }
]

Do not duplicate measurements or create multiple area calculations.
"""
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": enhanced_building_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }]
        }
        
        try:
            timeout = 60 if CLOUD_MODE else 90
            response = requests.post(GEMINI_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    measurements = self._extract_building_measurements(content)
                    measurements = self._enhanced_area_calculation(measurements)
                    
                    return {
                        'success': True,
                        'measurements': measurements,
                        'raw_response': content
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_building_measurements(self, response: str) -> List[Dict]:
        measurements = []
    
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                raw_data = json.loads(json_content)
            
                for item in raw_data:
                    # Add None checks for all values
                    text_val = item.get('text', '')
                    value_val = item.get('value')
                    unit_val = item.get('unit', 'FT')
                    type_val = item.get('type', 'unknown')
                    component_val = item.get('component', 'unspecified')
                
                    # Skip if value is None or not convertible to float
                    if value_val is None:
                        continue
                
                    try:
                        value_float = float(value_val)
                        if value_float <= 0:  # Skip zero or negative values
                            continue
                    except (ValueError, TypeError):
                        continue
                
                    # Only include dwelling-related measurements
                    if type_val and 'dwelling' in str(type_val).lower():
                        measurement = {
                            'text': str(text_val) if text_val is not None else '',
                            'value': value_float,
                            'unit': str(unit_val) if unit_val is not None else 'FT',
                            'type': str(type_val) if type_val is not None else 'unknown',
                            'component': str(component_val) if component_val is not None else 'unspecified'
                        }
                        measurements.append(measurement)
            
                if measurements:
                    return measurements
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Log the error for debugging
            st.warning(f"JSON parsing failed for building measurements: {str(e)}")
    
        return self._enhanced_building_extraction(response)
    
    def _enhanced_building_extraction(self, text: str) -> List[Dict]:
        measurements = []
        
        area_patterns = [
            (r'(?:BUILDING|DWELLING|FLOOR)\s*(?:AREA\s*[=:]?\s*)?(\d{1,5}[,\s]*\d*)\s*S\.?F\.?', 'SF', 'dwelling_area'),
            (r'(?:FOOTPRINT[\s:=]*)?(\d{1,5}[,\s]*\d*)\s*S\.?F\.?\s*(?:FOOTPRINT)?', 'SF', 'dwelling_area'),
            (r'(\d{1,5}[,\s]*\d*)\s*SQ\.?\s*FT\.?\s*(?:FOOTPRINT|BUILDING|DWELLING)', 'SF', 'dwelling_area')
        ]
        
        dimension_patterns = [
            (r'(\d{1,2}\.?\d{0,2})\s*FT\.?\s*(?:WIDTH|WIDE)', 'FT', 'dwelling_width'),
            (r'(\d{1,2}\.?\d{0,2})\s*FT\.?\s*(?:LENGTH|LONG|DEPTH)', 'FT', 'dwelling_length'),
            (r'(\d{1,2}\.?\d{0,2})\s*FT\.?', 'FT', 'building_dimension'),
            (r"(\d{1,2}\.?\d{0,2})'", 'FT', 'building_dimension')
        ]
        
        height_patterns = [
            (r'(\d+\.?\d*)\s*STORY', 'stories', 'dwelling_height'),
            (r'(\d+\.?\d*)\s*FT\.?\s*(?:HEIGHT|HIGH|TALL)', 'FT', 'dwelling_height')
        ]
        
        all_patterns = area_patterns + dimension_patterns + height_patterns
        
        for pattern, unit, default_type in all_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if unit == 'stories':
                        if '1½' in match.group(0):
                            value = 1.5
                        else:
                            value_str = match.group(1)
                            value = float(value_str)
                    else:
                        value_str = match.group(1).replace(',', '').replace(' ', '').replace("'", '').replace('"', '')
                        value = float(value_str)
                    
                    if default_type == 'building_dimension':
                        if 500 <= value <= 5000:
                            default_type = 'dwelling_area'
                        elif 20 <= value <= 80:
                            default_type = 'dwelling_width'
                        elif 5 <= value <= 25:
                            default_type = 'dwelling_component_width'
                        else:
                            continue
                    
                    measurement = {
                        'text': match.group(0).strip(),
                        'value': value,
                        'unit': unit,
                        'type': default_type,
                        'method': 'enhanced_pattern_matching'
                    }
                    measurements.append(measurement)
                    
                except (ValueError, IndexError, TypeError):
                    continue
        
        return measurements
    
    def _enhanced_area_calculation(self, measurements: List[Dict]) -> List[Dict]:
        has_explicit_area = any(m['type'] == 'dwelling_area' for m in measurements)
        
        widths = [m for m in measurements if 'width' in m['type'] and 'component' not in m['type']]
        lengths = [m for m in measurements if 'length' in m['type'] and 'component' not in m['type']]
        
        if widths and lengths:
            main_width = max(widths, key=lambda x: x['value'])
            main_length = max(lengths, key=lambda x: x['value'])
            
            main_area = main_width['value'] * main_length['value']
            
            calculated_measurement = {
                'text': f"Calculated from {main_width['value']:.2f}' × {main_length['value']:.2f}'",
                'value': main_area,
                'unit': 'SF',
                'type': 'dwelling_area_calculated',
                'method': 'main_dimensions_calculation'
            }
            
            if not has_explicit_area:
                measurements.append(calculated_measurement)
            else:
                calculated_measurement['type'] = 'dwelling_area_verification'
                measurements.append(calculated_measurement)
        
        return measurements

class AddressExtractor:
    """Extract property address from architectural drawings"""
    
    def extract_address_from_drawing(self, image_path_or_pil) -> Dict[str, Any]:
        try:
            # Same image encoding logic as your existing detectors
            if isinstance(image_path_or_pil, (str, Path)):
                with open(image_path_or_pil, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                img_buffer = io.BytesIO()
                if CLOUD_MODE:
                    w, h = image_path_or_pil.size
                    if w > 1500 or h > 1500:
                        scale = min(1500/w, 1500/h)
                        new_size = (int(w*scale), int(h*scale))
                        image_path_or_pil = image_path_or_pil.resize(new_size, Image.Resampling.LANCZOS)
                
                image_path_or_pil.save(img_buffer, format='PNG', optimize=True)
                img_buffer.seek(0)
                base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Failed to encode image: {e}'}
        
        address_prompt = """
        EXTRACT PROPERTY ADDRESS FROM ARCHITECTURAL DRAWING
        
        Look for address information in these formats:
        - "PROPERTY ADDRESS:", "SITE ADDRESS:", "LOT ADDRESS:"
        - Street numbers followed by street names
        - Address in title blocks or information panels
        - "123 Main Street", "45 Oak Avenue", etc.
        - Look for Livingston Township addresses specifically
        
        Return JSON format:
        {
            "street_number": "123",
            "street_name": "Main Street", 
            "full_address": "123 Main Street",
            "township": "Livingston",
            "confidence": "high",
            "found": true
        }
        
        If no address found, return: {"found": false, "error": "No address detected"}
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": address_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }]
        }
        
        try:
            timeout = 60 if CLOUD_MODE else 90
            response = requests.post(GEMINI_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    address_data = self._extract_address_data(content)
                    return {
                        'success': True,
                        'address_data': address_data,
                        'raw_response': content
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_address_data(self, response: str) -> Dict:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        
        # Fallback extraction
        import re
        address_patterns = [
            r'(\d+)\s+([A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct))',
            r'PROPERTY\s+ADDRESS[:\s]+(.+)',
            r'SITE\s+ADDRESS[:\s]+(.+)'
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:  # Street number and name
                    return {
                        'street_number': match.group(1),
                        'street_name': match.group(2).strip(),
                        'full_address': f"{match.group(1)} {match.group(2).strip()}",
                        'found': True,
                        'confidence': 'medium'
                    }
                else:  # Full address
                    full_address = match.group(1).strip()
                    return {
                        'full_address': full_address,
                        'found': True,
                        'confidence': 'medium'
                    }
        
        return {'found': False, 'error': 'No address pattern matched'}

class TaxMapLotFinder:
    """Find specific lot dimensions from tax map PDFs"""
    
    def __init__(self, tax_map_pdfs_directory):
        self.tax_maps_dir = tax_map_pdfs_directory
    
    def find_lot_by_address(self, address_data) -> Dict[str, Any]:
        """Find the lot in tax maps using address"""
        if not address_data.get('found'):
            return {'success': False, 'error': 'No valid address provided'}
        
        # For Livingston Township, we know the structure from your PDFs
        # The tax map you provided is Sheet 16, so we'll search through available sheets
        
        full_address = address_data.get('full_address', '')
        street_name = address_data.get('street_name', '')
        
        # Search through tax map files to find the one containing this address
        tax_map_files = self._get_tax_map_files()
        
        for tax_map_file in tax_map_files:
            lot_info = self._search_address_in_tax_map(tax_map_file, full_address, street_name)
            if lot_info.get('success'):
                return lot_info
        
        return {'success': False, 'error': 'Address not found in tax maps'}
    
    def _get_tax_map_files(self):
        """Get list of tax map PDF files"""
        import os
        tax_map_files = []
        if os.path.exists(self.tax_maps_dir):
            for file in os.listdir(self.tax_maps_dir):
                if file.endswith('.pdf'):
                    tax_map_files.append(os.path.join(self.tax_maps_dir, file))
        return tax_map_files
    
    def _search_address_in_tax_map(self, tax_map_path, full_address, street_name):
        """Search for address in a specific tax map"""
        try:
            # Convert tax map PDF to image
            pdf_converter = PDFtoImageConverter(dpi=200)  # Lower DPI for faster processing
            images = pdf_converter.convert_pdf_to_images(tax_map_path)
            
            if not images:
                return {'success': False, 'error': 'Could not convert tax map'}
            
            # Use first page (most tax maps are single page)
            tax_map_image = images[0]
            
            # Search for the address/street in the tax map
            search_result = self._find_address_in_image(tax_map_image, full_address, street_name)
            
            if search_result.get('success'):
                # Extract lot dimensions from the same image
                lot_dimensions = self._extract_lot_dimensions_from_tax_map(
                    tax_map_image, 
                    search_result.get('block_number'), 
                    search_result.get('lot_number')
                )
                
                return {
                    'success': True,
                    'tax_map_file': tax_map_path,
                    'block_number': search_result.get('block_number'),
                    'lot_number': search_result.get('lot_number'),
                    'official_dimensions': lot_dimensions,
                    'sheet_info': search_result.get('sheet_info')
                }
            
            return {'success': False, 'error': 'Address not found in this tax map'}
            
        except Exception as e:
            return {'success': False, 'error': f'Error processing tax map: {str(e)}'}
    
    def _find_address_in_image(self, tax_map_image, full_address, street_name):
        """Use Gemini to find address in tax map image"""
        try:
            img_buffer = io.BytesIO()
            tax_map_image.save(img_buffer, format='PNG', optimize=True)
            img_buffer.seek(0)
            base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Image encoding failed: {e}'}
        
        address_search_prompt = f"""
        FIND ADDRESS IN TAX MAP
        
        Search for address: "{full_address}" or street: "{street_name}"
        
        In this tax map, locate:
        1. The street name that matches "{street_name}"
        2. Find the specific lot number for address "{full_address}"
        3. Identify the block number containing this lot
        4. Note any sheet reference numbers
        
        Look for:
        - Street names in the map
        - Lot numbers (small numbers inside parcels)
        - Block numbers (larger numbers, often in circles or boxes)
        - Address ranges along streets
        
        Return JSON:
        {{
            "found": true/false,
            "block_number": "16",
            "lot_number": "45", 
            "street_found": "Bryant Drive",
            "sheet_info": "Sheet 16",
            "confidence": "high"
        }}
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": address_search_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }]
        }
        
        try:
            timeout = 90 if CLOUD_MODE else 120
            response = requests.post(GEMINI_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_address_search_result(content)
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_lot_dimensions_from_tax_map(self, tax_map_image, block_number, lot_number):
        """Extract official lot dimensions from tax map"""
        try:
            img_buffer = io.BytesIO()
            tax_map_image.save(img_buffer, format='PNG', optimize=True)
            img_buffer.seek(0)
            base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Image encoding failed: {e}'}
        
        dimensions_prompt = f"""
        EXTRACT OFFICIAL LOT DIMENSIONS FROM TAX MAP
        
        Focus on Block {block_number}, Lot {lot_number}
        
        Find ALL official measurements for this specific lot:
        1. Lot area (square feet) - often shown as "XXXX SF" or "X.XX AC"
        2. Lot frontage/width (feet) - dimension along street
        3. Lot depth (feet) - dimension perpendicular to street  
        4. All boundary dimensions around the lot perimeter
        5. Any setback requirements if noted
        
        Look for:
        - Dimensions with ' or " marks (feet/inches)
        - Area measurements with "SF", "SQ FT", or "AC" 
        - Numbers along lot boundary lines
        - Measurements inside or adjacent to the lot parcel
        
        Return JSON with all found measurements:
        {{
            "lot_area": {{"value": 8500, "unit": "SF"}},
            "lot_width": {{"value": 75, "unit": "FT"}}, 
            "lot_depth": {{"value": 120, "unit": "FT"}},
            "frontage": {{"value": 75, "unit": "FT"}},
            "all_dimensions": [
                {{"measurement": "75'", "type": "width", "value": 75}},
                {{"measurement": "120'", "type": "depth", "value": 120}}
            ]
        }}
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": dimensions_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }]
        }
        
        try:
            timeout = 90 if CLOUD_MODE else 120
            response = requests.post(GEMINI_URL, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return self._parse_lot_dimensions(content)
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_address_search_result(self, response: str) -> Dict:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                result = json.loads(json_content)
                result['success'] = result.get('found', False)
                return result
        except json.JSONDecodeError:
            pass
        return {'success': False, 'error': 'Could not parse search result'}
    
    def _parse_lot_dimensions(self, response: str) -> Dict:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response[json_start:json_end]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        return {'success': False, 'error': 'Could not parse dimensions'}

class MeasurementVerifier:
    """Compare extracted measurements with official tax map data"""
    
    def verify_lot_measurements(self, extracted_data, official_data):
        """Compare lot measurements with tolerance"""
        if not official_data or not extracted_data:
            return {'success': False, 'error': 'Missing data for comparison'}
        
        verification_results = {}
        
        # Compare lot area
        if official_data.get('lot_area') and extracted_data.get('lot_measurements'):
            official_area = official_data['lot_area'].get('value')
            extracted_areas = [m for m in extracted_data['lot_measurements'] if 'area' in m['type']]
            
            if extracted_areas and official_area:
                extracted_area = extracted_areas[0]['value']
                verification_results['lot_area'] = self._compare_values(
                    extracted_area, official_area, tolerance_pct=5
                )
        
        # Compare lot width
        if official_data.get('lot_width') and extracted_data.get('lot_measurements'):
            official_width = official_data['lot_width'].get('value')
            extracted_widths = [m for m in extracted_data['lot_measurements'] if 'width' in m['type']]
            
            if extracted_widths and official_width:
                extracted_width = extracted_widths[0]['value']
                verification_results['lot_width'] = self._compare_values(
                    extracted_width, official_width, tolerance_pct=3
                )
        
        # Compare lot depth
        if official_data.get('lot_depth') and extracted_data.get('lot_measurements'):
            official_depth = official_data['lot_depth'].get('value')
            extracted_depths = [m for m in extracted_data['lot_measurements'] if 'depth' in m['type']]
            
            if extracted_depths and official_depth:
                extracted_depth = extracted_depths[0]['value']
                verification_results['lot_depth'] = self._compare_values(
                    extracted_depth, official_depth, tolerance_pct=3
                )
        
        return {
            'success': True,
            'verification_results': verification_results,
            'official_source': 'Livingston Township Tax Maps',
            'verification_date': datetime.now().isoformat()
        }
    
    def _compare_values(self, extracted, official, tolerance_pct):
        """Compare individual measurements with tolerance"""
        if not extracted or not official:
            return {'status': 'MISSING_DATA', 'confidence': 'low'}
        
        difference_pct = abs(extracted - official) / official * 100
        difference_abs = abs(extracted - official)
        
        if difference_pct <= tolerance_pct:
            status = 'VERIFIED'
            confidence = 'high'
        elif difference_pct <= tolerance_pct * 2:
            status = 'ACCEPTABLE_VARIANCE'
            confidence = 'medium'
        else:
            status = 'SIGNIFICANT_DISCREPANCY'
            confidence = 'low'
        
        return {
            'status': status,
            'confidence': confidence,
            'extracted_value': extracted,
            'official_value': official,
            'difference_pct': difference_pct,
            'difference_abs': difference_abs,
            'tolerance_pct': tolerance_pct
        }

class ZoningTableCompiler:
    """Compile comprehensive zoning analysis from all measurements using Livingston Township requirements"""
    
    def __init__(self):
        self.zoning_requirements = LIVINGSTON_ZONING_REQUIREMENTS
        self.default_zone = 'R-4'  # Default if no zone detected
    
    def detect_zone_from_yolo_results(self, yolo_results):
        """Detect zone designation from YOLO detection results"""
        detected_zones = []
        
        if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # This would depend on your YOLO model's class names
                # For now, we'll check if any detections suggest zoning information
                detected_zones.append({
                    'zone': 'ZONE_DETECTED',
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detected_zones
    
    def determine_zone_from_context(self, lot_data, setback_data, building_data, yolo_detection_summary):
        """Assume all architectural drawings are R-4 zone for simplified testing"""
        # For now, always return R-4 to focus on measurement extraction
        return 'R-4'
    
    def compile_zoning_analysis(self, lot_data, setback_data, building_data, image_name, yolo_detection_summary=None):
        """Compile comprehensive zoning analysis with automatic zone detection"""
        
        # Determine the applicable zone
        detected_zone = self.determine_zone_from_context(lot_data, setback_data, building_data, yolo_detection_summary or {})
        zone_requirements = self.zoning_requirements.get(detected_zone, self.zoning_requirements[self.default_zone])
        
        analysis = {
            'source_image': image_name,
            'timestamp': datetime.now().isoformat(),
            'detected_zone': detected_zone,
            'zone_requirements': zone_requirements,
            'lot_measurements': lot_data.get('measurements', []) if lot_data.get('success') else [],
            'setback_measurements': setback_data.get('measurements', []) if setback_data.get('success') else [],
            'building_measurements': building_data.get('measurements', []) if building_data.get('success') else [],
            'zoning_table': {},
            'compliance_analysis': {},
            'summary': {}
        }
        
        # Process measurements
        lot_summary = self._process_lot_measurements(analysis['lot_measurements'])
        setback_summary = self._process_setback_measurements(analysis['setback_measurements'])
        building_summary = self._process_building_measurements(analysis['building_measurements'])
        
        # Build zoning table using detected zone requirements
        analysis['zoning_table'] = {
            'MIN_LOT_AREA': self._format_measurement_with_requirement(
                lot_summary.get('area'), zone_requirements.get('min_lot_area'), '>='),
            'MIN_LOT_WIDTH': self._format_measurement_with_requirement(
                lot_summary.get('min_width'), zone_requirements.get('min_lot_width'), '>='),
            'MIN_FRONT_SETBACK': self._format_measurement_with_requirement(
                setback_summary.get('front'), zone_requirements.get('front_setback'), '>='),
            'MIN_SIDE_SETBACK': self._format_measurement_with_requirement(
                setback_summary.get('min_side'), zone_requirements.get('side_setback'), '>='),
            'MIN_REAR_SETBACK': self._format_measurement_with_requirement(
                setback_summary.get('rear'), zone_requirements.get('rear_setback'), '>='),
            'AGGREGATE_SIDE_YARD': self._format_aggregate_side_yard(setback_summary, zone_requirements),
            'MAX_BUILDING_HEIGHT': self._format_measurement_with_requirement(
                building_summary.get('height'), zone_requirements.get('height'), '<='),
            'MAX_HF_AREA': self._format_measurement_with_requirement(
                building_summary.get('area'), zone_requirements.get('hf_area'), '<='),
            'MAX_HF_RATIO': self._calculate_hf_ratio_with_requirement(
                lot_summary.get('area'), building_summary.get('area'), zone_requirements.get('hf_ratio'))
        }
        
        # Compliance analysis using detected zone
        analysis['compliance_analysis'] = self._analyze_compliance_for_zone(analysis['zoning_table'], detected_zone)
        
        # Summary
        total_measurements = len(analysis['lot_measurements']) + len(analysis['setback_measurements']) + len(analysis['building_measurements'])
        analysis['summary'] = {
            'total_measurements': total_measurements,
            'lot_measurements_found': len(analysis['lot_measurements']),
            'setback_measurements_found': len(analysis['setback_measurements']),
            'building_measurements_found': len(analysis['building_measurements']),
            'completeness_score': self._calculate_completeness_score(analysis['zoning_table']),
            'overall_compliance': analysis['compliance_analysis'].get('overall_status', 'UNKNOWN'),
            'detected_zone': detected_zone,
            'zone_use': zone_requirements.get('use', 'Unknown')
        }
        
        return analysis
    
    def _format_measurement_with_requirement(self, measurement, requirement, operator):
        """Format measurement with zone-specific requirement checking"""
        if not measurement:
            req_text = f" (Req: {requirement})" if requirement is not None else " (No req.)"
            return {
                'value': 'N/A' + req_text, 
                'compliance': 'UNKNOWN', 
                'variance_needed': 'N/A', 
                'requirement': requirement
            }
        
        if measurement['unit'] == 'SF':
            formatted = f"{measurement['value']:,.0f} SF"
        elif measurement['unit'] == 'stories':
            formatted = f"{measurement['value']} {'story' if measurement['value'] == 1 else 'stories'}"
        else:
            formatted = f"{measurement['value']:.2f}'"
        
        # Check compliance
        compliance = 'UNKNOWN'
        variance_needed = 'N/A'
        
        if requirement is not None:
            if operator == '>=' and measurement['value'] >= requirement:
                compliance = 'COMPLIANT'
                variance_needed = 'NO'
            elif operator == '<=' and measurement['value'] <= requirement:
                compliance = 'COMPLIANT'
                variance_needed = 'NO'
            elif operator == '>=' and measurement['value'] < requirement:
                compliance = 'NON-COMPLIANT'
                variance_needed = 'YES'
            elif operator == '<=' and measurement['value'] > requirement:
                compliance = 'NON-COMPLIANT'
                variance_needed = 'YES'
        else:
            # No requirement for this zone
            compliance = 'N/A'
            variance_needed = 'N/A'
        
        req_text = f" (Req: {requirement})" if requirement is not None else " (No req.)"
        
        return {
            'value': formatted + req_text,
            'raw_value': measurement['value'],
            'compliance': compliance,
            'variance_needed': variance_needed,
            'requirement': requirement
        }
    
    def _format_aggregate_side_yard(self, setback_summary, zone_requirements):
        """Format aggregate side yard with 30% rule checking"""
        aggregate = setback_summary.get('aggregate')
        if not aggregate:
            return {'value': 'N/A', 'compliance': 'UNKNOWN', 'variance_needed': 'N/A'}
        
        # Handle None values in zone requirements
        side_setback_req = zone_requirements.get('side_setback')
        if side_setback_req is None:
            # For business zones that don't have side setback requirements
            return {
                'value': f"{aggregate['value']:.2f}' (No req. for this zone)",
                'raw_value': aggregate['value'],
                'compliance': 'N/A',
                'variance_needed': 'N/A',
                'requirement': 'N/A'
            }
        
        # Aggregate side yard requirement: typically minimum side setback * 2
        min_aggregate = side_setback_req * 2
        
        compliance = 'COMPLIANT' if aggregate['value'] >= min_aggregate else 'NON-COMPLIANT'
        variance_needed = 'NO' if compliance == 'COMPLIANT' else 'YES'
        
        return {
            'value': f"{aggregate['value']:.2f}' (Min: {min_aggregate}')",
            'raw_value': aggregate['value'],
            'compliance': compliance,
            'variance_needed': variance_needed,
            'requirement': f"≥{min_aggregate}'"
        }
    
    def _calculate_hf_ratio_with_requirement(self, lot_area, building_area, max_ratio):
        """Calculate HF (building coverage) ratio with zone-specific requirements"""
        if not lot_area or not building_area:
            return {'value': 'N/A', 'compliance': 'UNKNOWN', 'variance_needed': 'N/A'}
        
        ratio_pct = (building_area['value'] / lot_area['value']) * 100
        
        if max_ratio is None:
            # No coverage requirement for this zone
            return {
                'value': f"{ratio_pct:.1f}% (No limit for this zone)",
                'raw_value': ratio_pct,
                'compliance': 'N/A',
                'variance_needed': 'N/A',
                'requirement': None
            }
        
        compliance = 'COMPLIANT' if ratio_pct <= max_ratio else 'NON-COMPLIANT'
        variance_needed = 'NO' if compliance == 'COMPLIANT' else 'YES'
        
        return {
            'value': f"{ratio_pct:.1f}% (Max: {max_ratio}%)",
            'raw_value': ratio_pct,
            'compliance': compliance,
            'variance_needed': variance_needed,
            'requirement': max_ratio
        }
    
    def _analyze_compliance_for_zone(self, zoning_table, zone):
        """Analyze compliance for specific detected zone"""
        analysis = {}
        
        compliant_count = 0
        total_count = 0
        
        for requirement_name, data in zoning_table.items():
            if data.get('compliance') != 'UNKNOWN':
                total_count += 1
                if data.get('compliance') == 'COMPLIANT':
                    compliant_count += 1
                analysis[requirement_name] = data.get('compliance')
        
        if total_count > 0:
            if compliant_count == total_count:
                analysis['overall_status'] = 'FULLY_COMPLIANT'
            elif compliant_count > total_count / 2:
                analysis['overall_status'] = 'MOSTLY_COMPLIANT'
            else:
                analysis['overall_status'] = 'NON_COMPLIANT'
        else:
            analysis['overall_status'] = 'INSUFFICIENT_DATA'
        
        analysis['compliance_rate'] = f"{compliant_count}/{total_count}" if total_count > 0 else "0/0"
        analysis['applicable_zone'] = zone
        
        return analysis
    
    def _process_lot_measurements(self, measurements):
        summary = {}
        
        # Find area
        areas = [m for m in measurements if 'area' in m['type']]
        if areas:
            summary['area'] = max(areas, key=lambda x: x['value'])
        
        # Find widths and depths
        widths = [m for m in measurements if m['type'] in ['lot_width', 'lot_frontage']]
        depths = [m for m in measurements if m['type'] == 'lot_depth']
        
        if widths:
            summary['min_width'] = min(widths, key=lambda x: x['value'])
        if depths:
            summary['min_depth'] = min(depths, key=lambda x: x['value'])
        
        return summary
    
    # In ZoningTableCompiler class, replace _process_setback_measurements
    def _process_setback_measurements(self, measurements):
        summary = {}
    
        # Group setbacks by type
        setback_groups = {
            'front_setback': [],
            'rear_setback': [],
            'side_setback': [],
            'left_side_setback': [],
            'right_side_setback': []
        }
    
        for m in measurements:
            setback_type = m['type']
            if setback_type in setback_groups:
                setback_groups[setback_type].append(m)
            elif 'setback' in setback_type:
                setback_groups['side_setback'].append(m)
    
        # Process each group
        for group_name, setbacks in setback_groups.items():
            if setbacks:
                if group_name == 'front_setback':
                    summary['front'] = setbacks[0]  # Take first front setback
                elif group_name == 'rear_setback':
                    summary['rear'] = setbacks[0]   # Take first rear setback
                elif group_name in ['side_setback', 'left_side_setback', 'right_side_setback']:
                    # Handle multiple side setbacks
                    for i, setback in enumerate(setbacks):
                        summary[f'side_{i+1}'] = setback
    
        # Calculate minimum side and aggregate from all side setbacks
        all_sides = [v for k, v in summary.items() if k.startswith('side_')]
        if all_sides:
            summary['min_side'] = min(all_sides, key=lambda x: x['value'])
            if len(all_sides) >= 2:
                total_aggregate = sum(s['value'] for s in all_sides)
                summary['aggregate'] = {
                    'value': total_aggregate,
                    'unit': 'FT',
                    'text': f"Total aggregate: {total_aggregate:.2f}'"
                }
    
        return summary
    
    def _process_building_measurements(self, measurements):
        summary = {}
        
        # Find area
        areas = [m for m in measurements if 'area' in m['type']]
        if areas:
            explicit_areas = [a for a in areas if 'calculated' not in a['type']]
            summary['area'] = explicit_areas[0] if explicit_areas else areas[0]
        
        # Find height
        heights = [m for m in measurements if 'height' in m['type']]
        if heights:
            summary['height'] = heights[0]
        
        return summary
    
    def _calculate_completeness_score(self, zoning_table):
        total_fields = len(zoning_table)
        filled_fields = sum(1 for item in zoning_table.values() if item.get('value') != 'N/A')
        return int((filled_fields / total_fields) * 100) if total_fields > 0 else 0

def create_comprehensive_zip(analysis_results, extracted_data, pdf_filename):
    """Create comprehensive ZIP with all analysis results"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add extracted YOLO regions
        if extracted_data:
            for filename, image in extracted_data['cropped_images'].items():
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                zip_file.writestr(f"yolo_detections/{filename}.png", img_buffer.getvalue())
        
        # Add measurement analysis results
        if analysis_results:
            # Comprehensive analysis JSON
            zip_file.writestr("analysis_results.json", json.dumps(analysis_results, indent=2, default=str))
            
            # Zoning table CSV
            if analysis_results.get('zoning_table'):
                csv_data = []
                for requirement, data in analysis_results['zoning_table'].items():
                    csv_data.append({
                        'Requirement': requirement,
                        'Value': data.get('value', 'N/A'),
                        'Compliance': data.get('compliance', 'UNKNOWN'),
                        'Variance_Needed': data.get('variance_needed', 'N/A')
                    })
                
                df = pd.DataFrame(csv_data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr("zoning_table.csv", csv_buffer.getvalue())
            
            # Detailed measurements
            all_measurements = []
            for measurement_type in ['lot_measurements', 'setback_measurements', 'building_measurements']:
                measurements = analysis_results.get(measurement_type, [])
                for m in measurements:
                    m['measurement_category'] = measurement_type
                    all_measurements.append(m)
            
            if all_measurements:
                df_measurements = pd.DataFrame(all_measurements)
                csv_buffer = io.StringIO()
                df_measurements.to_csv(csv_buffer, index=False)
                zip_file.writestr("all_measurements.csv", csv_buffer.getvalue())
        
        # Summary report
        summary_text = f"""COMPREHENSIVE ARCHITECTURAL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source PDF: {pdf_filename}

=== ANALYSIS SUMMARY ===
"""
        
        if analysis_results:
            summary = analysis_results.get('summary', {})
            summary_text += f"""
Total Measurements Found: {summary.get('total_measurements', 0)}
- Lot Measurements: {summary.get('lot_measurements_found', 0)}
- Setback Measurements: {summary.get('setback_measurements_found', 0)}
- Building Measurements: {summary.get('building_measurements_found', 0)}

Completeness Score: {summary.get('completeness_score', 0)}%
Overall Compliance: {summary.get('overall_compliance', 'UNKNOWN')}

=== ZONING REQUIREMENTS ANALYSIS ===
"""
            
            zoning_table = analysis_results.get('zoning_table', {})
            for req, data in zoning_table.items():
                compliance = data.get('compliance', 'UNKNOWN')
                variance = data.get('variance_needed', 'N/A')
                summary_text += f"{req}: {data.get('value', 'N/A')} - {compliance} (Variance: {variance})\n"
        
        if extracted_data:
            summary_text += f"""

=== YOLO DETECTION SUMMARY ===
Total YOLO Detections: {extracted_data['summary']['total_extractions']}

Detected Classes:
"""
            for class_name, count in extracted_data['summary']['regions_by_class'].items():
                summary_text += f"- {class_name}: {count}\n"
        
        zip_file.writestr("comprehensive_report.txt", summary_text)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def display_zoning_analysis_dashboard(analysis_results):
    """Display comprehensive zoning analysis dashboard"""
    if not analysis_results:
        st.warning("No analysis results to display")
        return
    
    st.header("Comprehensive Zoning Analysis Dashboard")
    
    # Summary metrics
    summary = analysis_results.get('summary', {})
    compliance = analysis_results.get('compliance_analysis', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Measurements", summary.get('total_measurements', 0))
    
    with col2:
        st.metric("Completeness Score", f"{summary.get('completeness_score', 0)}%")
    
    with col3:
        overall_compliance = summary.get('overall_compliance', 'UNKNOWN')
        st.metric("Overall Compliance", overall_compliance)
    
    with col4:
        compliance_rate = compliance.get('compliance_rate', '0/0')
        st.metric("Compliance Rate", compliance_rate)
    
    # Zoning Requirements Table
    st.subheader("Zoning Requirements Analysis")
    
    zoning_table = analysis_results.get('zoning_table', {})
    if zoning_table:
        # Create DataFrame for display
        table_data = []
        for requirement, data in zoning_table.items():
            table_data.append({
                'Requirement': requirement.replace('_', ' ').title(),
                'Value': data.get('value', 'N/A'),
                'Compliance': data.get('compliance', 'UNKNOWN'),
                'Variance Needed': data.get('variance_needed', 'N/A')
            })
        
        df = pd.DataFrame(table_data)
        
        # Style the dataframe
        def style_compliance(val):
            if val == 'COMPLIANT':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'NON-COMPLIANT':
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        styled_df = df.style.applymap(style_compliance, subset=['Compliance'])
        st.dataframe(styled_df, use_container_width=True)
    
    # Detailed Measurements
    tabs = st.tabs(["Lot Measurements", "Setback Measurements", "Building Measurements"])
    
    with tabs[0]:
        lot_measurements = analysis_results.get('lot_measurements', [])
        if lot_measurements:
            st.write(f"Found {len(lot_measurements)} lot measurements:")
            for i, m in enumerate(lot_measurements, 1):
                st.write(f"{i}. **{m['text']}** = {m['value']} {m['unit']} ({m['type']})")
        else:
            st.write("No lot measurements found")
    
    with tabs[1]:
        setback_measurements = analysis_results.get('setback_measurements', [])
        if setback_measurements:
            st.write(f"Found {len(setback_measurements)} setback measurements:")
            for i, m in enumerate(setback_measurements, 1):
                confidence = m.get('confidence', 'medium')
                st.write(f"{i}. **{m['text']}** = {m['value']} {m['unit']} ({m['type']}) - Confidence: {confidence}")
        else:
            st.write("No setback measurements found")
    
    with tabs[2]:
        building_measurements = analysis_results.get('building_measurements', [])
        if building_measurements:
            st.write(f"Found {len(building_measurements)} building measurements:")
            for i, m in enumerate(building_measurements, 1):
                component = m.get('component', 'unspecified')
                st.write(f"{i}. **{m['text']}** = {m['value']} {m['unit']} ({m['type']}) - Component: {component}")
        else:
            st.write("No building measurements found")

def main():
    st.set_page_config(
        page_title="Comprehensive Architectural Analysis Pipeline",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Comprehensive Architectural Analysis Pipeline")
    st.markdown("Upload a PDF → Extract architectural drawings → Analyze lot, setback, and building measurements → Generate zoning compliance report")
    
    # Cloud deployment notice
    if CLOUD_MODE:
        st.info("🌤️ **Cloud Mode Active**: Optimized for Streamlit Cloud with reduced memory usage and faster processing")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model loading with multiple path options
    yolo_pipeline = None
    try:
        yolo_pipeline = YOLOv8Pipeline()
        if yolo_pipeline.custom_model:
            st.sidebar.success("✅ Custom YOLO model loaded successfully")
        else:
            st.sidebar.info("ℹ️ Using default YOLOv8 model")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading YOLO model: {str(e)}")
        st.error("Failed to initialize YOLO model. Please check the logs.")
        return
    
    # Configuration parameters
    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.25, step=0.05)
    
    # Gemini API key check
    if GEMINI_KEY:
        st.sidebar.success("✅ Gemini AI configured for measurement analysis")
    else:
        st.sidebar.error("❌ Gemini API key not configured")
        st.sidebar.warning("Please set your Gemini API key in the code")
    
    # Cloud deployment settings display
    st.sidebar.markdown("**Cloud Optimizations:**")
    st.sidebar.info(f"PDF DPI: {PDF_DPI}")
    st.sidebar.info(f"Max Image Size: {MAX_IMAGE_DIMENSION}px")
    st.sidebar.info(f"Max File Size: {MAX_FILE_SIZE_MB}MB")
    
    # Initialize components
    pdf_converter = PDFtoImageConverter(dpi=PDF_DPI)
    validator = AnalysisValidator()
    lot_detector = EnhancedLotMeasurementsDetector()
    setback_detector = EnhancedSetbackMeasurementsDetector()
    building_detector = EnhancedBuildingMeasurementsDetector()
    zoning_compiler = ZoningTableCompiler()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload")
        uploaded_pdf = st.file_uploader(
            "Upload PDF file with architectural drawings",
            type=['pdf'],
            help=f"Upload a PDF containing architectural plans (max {MAX_FILE_SIZE_MB}MB)"
        )
        
        if uploaded_pdf:
            # Check file size
            file_size_mb = len(uploaded_pdf.getvalue()) / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"PDF too large ({file_size_mb:.1f}MB). Please use a smaller file (max {MAX_FILE_SIZE_MB}MB).")
                st.info("Try compressing the PDF or selecting fewer pages.")
                return
            
            st.success(f"PDF uploaded: {uploaded_pdf.name} ({file_size_mb:.1f}MB)")
            
            # Convert PDF to images with enhanced error handling
            try:
                with st.spinner("Converting PDF to images..."):
                    pdf_bytes = uploaded_pdf.getvalue()
                    images = pdf_converter.convert_pdf_to_images(pdf_bytes, max_dimension=MAX_IMAGE_DIMENSION)
            except Exception as e:
                st.error(f"PDF conversion failed: {str(e)}")
                st.info("Try uploading a smaller PDF or a single-page document.")
                if CLOUD_MODE:
                    st.info("Large PDFs may exceed cloud memory limits. Consider using a PDF with fewer pages.")
                return
            
            if images:
                st.success(f"Converted PDF to {len(images)} page(s)")
                
                # Page selection
                if len(images) > 1:
                    page_idx = st.selectbox("Select page to analyze:", range(len(images)), format_func=lambda x: f"Page {x+1}")
                else:
                    page_idx = 0
                
                selected_image = images[page_idx]
                
                # Display original image
                st.subheader("Original Image")
                st.image(selected_image, use_column_width=True)
                
                # Clear previous results when new image is selected
                if 'selected_page' not in st.session_state or st.session_state.selected_page != page_idx:
                    st.session_state.selected_page = page_idx
                    for key in ['detection_results', 'extracted_data', 'analysis_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                
                # Step 1: YOLO Detection
                if st.button("🔍 Step 1: Run YOLO Detection", type="primary"):
                    with col2:
                        try:
                            with st.spinner("Running YOLO detection..."):
                                annotated_image, results = yolo_pipeline.predict_image(selected_image, conf_threshold=conf_threshold)
                                
                                # Validate YOLO results
                                yolo_validation = validator.validate_step('yolo_completed', results)
                                
                                if not yolo_validation['passed']:
                                    st.warning("YOLO Detection Issues:")
                                    for issue in yolo_validation['issues']:
                                        st.warning(f"• {issue}")
                                    
                                    if yolo_validation['retry_needed']:
                                        st.info("Retrying with lower confidence threshold...")
                                        annotated_image, results = yolo_pipeline.predict_image(selected_image, conf_threshold=0.1)
                                        
                                        # Re-validate
                                        yolo_validation = validator.validate_step('yolo_completed', results)
                                        if yolo_validation['passed']:
                                            st.success("✅ Retry successful!")
                                        else:
                                            st.warning("⚠️ Proceeding with available detections")
                                
                                st.session_state.detection_results = results
                                st.session_state.selected_image = selected_image
                                st.session_state.annotated_image = annotated_image
                                st.session_state.pdf_name = uploaded_pdf.name
                                
                                st.header("🎯 YOLO Detection Results")
                                st.image(annotated_image, use_column_width=True)
                                
                                summary = yolo_pipeline.get_detection_summary(results)
                                st.subheader("Detection Summary")
                                col2_1, col2_2 = st.columns([1, 1])
                                
                                with col2_1:
                                    st.metric("Total Detections", summary['total_detections'])
                                    if summary['detections_per_class']:
                                        st.write("**Detections by Class:**")
                                        for class_name, count in summary['detections_per_class'].items():
                                            st.write(f"• {class_name}: {count}")
                                
                                with col2_2:
                                    if summary['detection_details']:
                                        df = pd.DataFrame(summary['detection_details'])
                                        df['confidence'] = df['confidence'].round(3)
                                        st.dataframe(df[['class', 'confidence']], use_container_width=True)
                                
                                # Show validation status
                                if yolo_validation['passed']:
                                    st.success("✅ YOLO detection completed successfully")
                                else:
                                    st.warning("⚠️ YOLO detection has validation issues but proceeding")
                        
                        except Exception as e:
                            st.error(f"YOLO detection failed: {str(e)}")
                            if CLOUD_MODE:
                                st.info("This may be due to cloud memory limits. Try a smaller image or lower confidence threshold.")
            else:
                st.error("No pages could be extracted from the PDF")
                if CLOUD_MODE:
                    st.info("This may be due to cloud processing limits. Try a smaller or simpler PDF.")
    
    # Step 2: Extract and Analyze (only show if YOLO detection is complete)
    if ('detection_results' in st.session_state and 
        st.session_state.detection_results is not None):
        
        st.markdown("---")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.header("🔬 Step 2: Extract & Analyze Measurements")
            
            if st.button("🚀 Extract Regions & Analyze Measurements", type="primary"):
                try:
                    # Extract detected regions
                    with st.spinner("Extracting detected regions..."):
                        extracted_data = yolo_pipeline.extract_detected_regions(
                            st.session_state.selected_image,
                            st.session_state.detection_results,
                            output_prefix=st.session_state.pdf_name.replace('.pdf', '')
                        )
                        
                        st.session_state.extracted_data = extracted_data
                    
                    if extracted_data['cropped_images']:
                        st.success(f"Extracted {len(extracted_data['cropped_images'])} regions")
                        
                        # Show preview of extracted regions
                        st.subheader("Extracted Regions Preview")
                        extracted_items = list(extracted_data['cropped_images'].items())
                        for i in range(0, min(6, len(extracted_items)), 3):  # Show max 6 images
                            cols = st.columns(3)
                            for j, (filename, img) in enumerate(extracted_items[i:i+3]):
                                with cols[j]:
                                    st.image(img, caption=filename.split('_')[1], use_column_width=True)
                        
                        # Run measurement analysis on suitable regions
                        with st.spinner("Analyzing measurements with Gemini AI..."):
                            # Select best region for analysis
                            best_region = None
                            best_filename = None
                            best_score = 0
                            
                            st.write("🔍 **Evaluating regions for architectural analysis:**")
                            
                            # Score each region for architectural content
                            for filename, image in extracted_data['cropped_images'].items():
                                score = 0
                                width, height = image.size
                                area = width * height
                                
                                st.write(f"• **{filename}**: {width}x{height} ({area:,} pixels)")
                                
                                # Prioritize architectural drawings over tables
                                if 'architecture' in filename.lower() or 'layout' in filename.lower():
                                    score += 1000
                                    st.write(f"  ✅ Architecture/Layout detected (+1000)")
                                elif 'site' in filename.lower() or 'plan' in filename.lower():
                                    score += 800
                                    st.write(f"  ✅ Site/Plan detected (+800)")
                                elif 'zone' in filename.lower() and 'map' not in filename.lower():
                                    score += 600
                                    st.write(f"  ✅ Zone drawing detected (+600)")
                                elif 'table' in filename.lower() or 'calculation' in filename.lower():
                                    score -= 500
                                    st.write(f"  ❌ Table/Calculation detected (-500)")
                                
                                # Size bonus for larger regions
                                if area > 200000:
                                    score += 200
                                    st.write(f"  ✅ Large region (+200)")
                                elif area > 100000:
                                    score += 100
                                    st.write(f"  ✅ Medium region (+100)")
                                
                                # Aspect ratio bonus
                                aspect_ratio = max(width, height) / min(width, height)
                                if 1.2 <= aspect_ratio <= 2.0:
                                    score += 50
                                    st.write(f"  ✅ Good aspect ratio (+50)")
                                
                                st.write(f"  **Final Score: {score}**")
                                
                                if score > best_score:
                                    best_score = score
                                    best_region = image
                                    best_filename = filename
                            
                            # Fallback selection if no good architectural region found
                            if best_score <= 0:
                                st.warning("No architectural drawings detected. Selecting largest region...")
                                best_filename, best_region = max(
                                    extracted_data['cropped_images'].items(),
                                    key=lambda x: x[1].size[0] * x[1].size[1]
                                )
                            
                            if best_region:
                                st.success(f"🎯 **Selected for analysis: {best_filename}** (Score: {best_score})")
                                
                                # Show the region being analyzed
                                st.subheader("Region Being Analyzed:")
                                st.image(best_region, caption=f"Analyzing: {best_filename}", use_column_width=True)
                                
                                # Run all three measurement analyses
                                st.write("🔍 **Running Measurement Analysis...**")
                                
                                lot_results = lot_detector.detect_lot_measurements(best_region)
                                if lot_results.get('success'):
                                    st.success(f"✅ Lot analysis: Found {len(lot_results.get('measurements', []))} measurements")
                                else:
                                    st.warning(f"⚠️ Lot analysis: {lot_results.get('error', 'No measurements found')}")
                                
                                setback_results = setback_detector.detect_setback_measurements(best_region)
                                if setback_results.get('success'):
                                    st.success(f"✅ Setback analysis: Found {len(setback_results.get('measurements', []))} measurements")
                                else:
                                    st.warning(f"⚠️ Setback analysis: {setback_results.get('error', 'No measurements found')}")
                                
                                building_results = building_detector.detect_building_measurements(best_region)
                                if building_results.get('success'):
                                    st.success(f"✅ Building analysis: Found {len(building_results.get('measurements', []))} measurements")
                                else:
                                    st.warning(f"⚠️ Building analysis: {building_results.get('error', 'No measurements found')}")
                                
                                # Show debug information if requested
                                if st.checkbox("Show Raw API Responses (Debug)", key="show_debug"):
                                    st.subheader("🔧 Debug Information")
                                    
                                    debug_tabs = st.tabs(["Lot Response", "Setback Response", "Building Response"])
                                    
                                    with debug_tabs[0]:
                                        if lot_results.get('raw_response'):
                                            st.text_area("Lot Analysis Response", lot_results['raw_response'][:1000], height=150)
                                    
                                    with debug_tabs[1]:
                                        if setback_results.get('raw_response'):
                                            st.text_area("Setback Analysis Response", setback_results['raw_response'][:1000], height=150)
                                    
                                    with debug_tabs[2]:
                                        if building_results.get('raw_response'):
                                            st.text_area("Building Analysis Response", building_results['raw_response'][:1000], height=150)
                                
                                # Compile comprehensive analysis
                                yolo_summary = yolo_pipeline.get_detection_summary(st.session_state.detection_results)
                                
                                analysis_results = zoning_compiler.compile_zoning_analysis(
                                    lot_results, setback_results, building_results, 
                                    best_filename, yolo_summary
                                )
                                
                                st.session_state.analysis_results = analysis_results

                                # NEW VERIFICATION SECTION
                                st.markdown("---")
                                st.header("🔍 Step 2.5: Official Tax Map Verification")
                
                                col_verify1, col_verify2 = st.columns([1, 1])
                
                                with col_verify1:
                                    st.markdown("**Tax Map Cross-Reference:**")
                    
                                    # Tax maps directory configuration
                                    tax_maps_directory = st.text_input(
                                        "Tax Maps Directory Path:", 
                                        value="tax_maps/",
                                        help="Directory containing Livingston Township tax map PDFs"
                                    )
                    
                                    if st.button("🗺️ Verify with Official Tax Maps", type="primary"):
                                        with st.spinner("Cross-referencing with official records..."):
                            
                                            # Extract address from architectural drawing
                                            address_extractor = AddressExtractor()
                                            address_result = address_extractor.extract_address_from_drawing(best_region)
                            
                                            if address_result.get('success') and address_result['address_data'].get('found'):
                                                address_data = address_result['address_data']
                                                st.success(f"📍 Address found: {address_data.get('full_address', 'N/A')}")
                                
                                                # Find lot in tax maps
                                                tax_map_finder = TaxMapLotFinder(tax_maps_directory)
                                                lot_result = tax_map_finder.find_lot_by_address(address_data)
                                
                                                if lot_result.get('success'):
                                                    st.success(f"🗺️ Found in tax map: Block {lot_result['block_number']}, Lot {lot_result['lot_number']}")
                                                    st.info(f"Source: {os.path.basename(lot_result['tax_map_file'])}")
                                    
                                                    # Verify measurements
                                                    verifier = MeasurementVerifier()
                                                    verification_result = verifier.verify_lot_measurements(
                                                        analysis_results, lot_result['official_dimensions']
                                                    )
                                    
                                                    if verification_result.get('success'):
                                                        st.session_state.verification_results = verification_result
                                                        st.session_state.official_dimensions = lot_result['official_dimensions']
                                        
                                                        with col_verify2:
                                                            st.markdown("**Verification Results:**")
                                            
                                                            verification_data = verification_result['verification_results']
                                            
                                                            for measurement, result in verification_data.items():
                                                                status = result['status']
                                                                if status == 'VERIFIED':
                                                                    st.success(f"✅ {measurement.replace('_', ' ').title()}: {status}")
                                                                elif status == 'ACCEPTABLE_VARIANCE':
                                                                    st.warning(f"⚠️ {measurement.replace('_', ' ').title()}: {status}")
                                                                else:
                                                                    st.error(f"❌ {measurement.replace('_', ' ').title()}: {status}")
                                                
                                                                st.write(f"   Extracted: {result['extracted_value']}")
                                                                st.write(f"   Official: {result['official_value']}")
                                                                st.write(f"   Difference: {result['difference_pct']:.1f}%")
                                                    else:
                                                        st.error(f"Verification failed: {verification_result.get('error')}")
                                                else:
                                                    st.warning(f"Could not locate lot in tax maps: {lot_result.get('error')}")
                                                    st.info("This may be due to address extraction issues or incomplete tax map database")
                                            else:
                                                st.warning(f"Address extraction failed: {address_result.get('error')}")
                                                st.info("Manual address input may be needed for tax map verification")
                    
                                    # Show debug info if available
                                    if st.checkbox("Show Address Extraction Debug", key="show_address_debug"):
                                        if 'address_result' in locals():
                                            st.text_area("Address Extraction Response", 
                                                address_result.get('raw_response', 'No response')[:500], 
                                                height=100)
                                
                                # Display results in the second column
                                with col4:
                                    st.header("📊 Measurement Analysis Results")
                                    
                                    # Show detected zone information
                                    detected_zone = analysis_results['summary'].get('detected_zone', 'R-4')
                                    zone_use = analysis_results['summary'].get('zone_use', 'Single Family')
                                    
                                    zone_col1, zone_col2 = st.columns(2)
                                    with zone_col1:
                                        st.metric("Detected Zone", detected_zone)
                                    with zone_col2:
                                        st.metric("Zone Use", zone_use)
                                    
                                    # Quick summary
                                    if analysis_results['summary']:
                                        summary = analysis_results['summary']
                                        metric_cols = st.columns(2)
                                        with metric_cols[0]:
                                            st.metric("Total Measurements", summary['total_measurements'])
                                        with metric_cols[1]:
                                            st.metric("Completeness", f"{summary['completeness_score']}%")
                                    
                                    # Show measurement counts
                                    measurement_cols = st.columns(3)
                                    with measurement_cols[0]:
                                        st.metric("Lot", len(analysis_results.get('lot_measurements', [])))
                                    with measurement_cols[1]:
                                        st.metric("Setback", len(analysis_results.get('setback_measurements', [])))
                                    with measurement_cols[2]:
                                        st.metric("Building", len(analysis_results.get('building_measurements', [])))
                                    
                                    # Show analysis status
                                    success_count = sum([
                                        lot_results.get('success', False),
                                        setback_results.get('success', False),
                                        building_results.get('success', False)
                                    ])
                                    
                                    if success_count == 3:
                                        st.success("✅ All measurement analyses completed successfully!")
                                    elif success_count > 0:
                                        st.warning(f"⚠️ {success_count}/3 measurement analyses completed")
                                    else:
                                        st.warning("⚠️ Limited measurement data available")
                                        if CLOUD_MODE:
                                            st.info("This may be due to image quality or cloud processing limitations")
                            else:
                                st.warning("No suitable regions found for measurement analysis")
                    else:
                        st.warning("No regions extracted from YOLO detection")
                        if CLOUD_MODE:
                            st.info("Try lowering the confidence threshold or using a clearer image")
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    if CLOUD_MODE:
                        st.info("This may be due to cloud memory limits or API timeouts. Try a smaller image or simpler PDF.")
    
    # Step 3: Results Dashboard (show if analysis exists)
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.markdown("---")
        
        analysis_results = st.session_state.analysis_results
        
        # Display comprehensive analysis results
        st.header("📊 Step 3: Comprehensive Analysis Results")
        
        # Show the architectural drawing being analyzed
        col_result1, col_result2 = st.columns([1, 2])
        
        with col_result1:
            st.markdown("**Analyzed Region:**")
            # Show the region that was analyzed
            if 'extracted_data' in st.session_state:
                extracted_images = st.session_state.extracted_data.get('cropped_images', {})
                source_image = analysis_results.get('source_image', '')
                if source_image in extracted_images:
                    st.image(extracted_images[source_image], 
                            caption=f"Source: {source_image}", 
                            use_column_width=True)
                else:
                    st.info("Analyzed region image not available")
            
            # Show detected zone and summary
            detected_zone = analysis_results.get('summary', {}).get('detected_zone', 'R-4')
            zone_use = analysis_results.get('summary', {}).get('zone_use', 'Single Family')
            
            st.metric("Detected Zone", detected_zone)
            st.metric("Zone Use", zone_use)
        
        with col_result2:
            st.markdown("**Analysis Summary:**")
            summary = analysis_results.get('summary', {})
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Measurements", summary.get('total_measurements', 0))
            with metric_cols[1]:
                st.metric("Completeness", f"{summary.get('completeness_score', 0)}%")
            with metric_cols[2]:
                st.metric("Overall Compliance", summary.get('overall_compliance', 'UNKNOWN'))
            with metric_cols[3]:
                compliance = analysis_results.get('compliance_analysis', {})
                compliance_rate = compliance.get('compliance_rate', '0/0')
                st.metric("Compliance Rate", compliance_rate)
        
        # Zoning Requirements Table
        st.markdown("**Zoning Requirements Analysis:**")
        
        zoning_table = analysis_results.get('zoning_table', {})
        if zoning_table:
            # Create DataFrame for display
            table_data = []
            for requirement, data in zoning_table.items():
                # Handle both dict and direct value cases
                if isinstance(data, dict):
                    value = data.get('value', 'N/A')
                    compliance = data.get('compliance', 'UNKNOWN')
                    variance = data.get('variance_needed', 'N/A')
                else:
                    value = str(data)
                    compliance = 'UNKNOWN'
                    variance = 'N/A'
                    
                table_data.append({
                    'Requirement': requirement.replace('_', ' ').title(),
                    'Value': value,
                    'Compliance': compliance,
                    'Variance Needed': variance
                })
            
            if table_data:
                df = pd.DataFrame(table_data)
                
                # Style the dataframe
                def style_compliance(val):
                    if val == 'COMPLIANT':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'NON-COMPLIANT':
                        return 'background-color: #f8d7da; color: #721c24'
                    else:
                        return 'background-color: #fff3cd; color: #856404'
                
                # Apply styling only to Compliance column if it exists
                if 'Compliance' in df.columns:
                    styled_df = df.style.applymap(style_compliance, subset=['Compliance'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No zoning table data available")
        else:
            st.warning("No zoning analysis results found")
        
        # Detailed measurements
        measurement_tabs = st.tabs(["📏 Lot Measurements", "📐 Setback Measurements", "🏠 Building Measurements"])
        
        with measurement_tabs[0]:
            lot_measurements = analysis_results.get('lot_measurements', [])
            if lot_measurements:
                st.write(f"**Found {len(lot_measurements)} lot measurements:**")
                for i, m in enumerate(lot_measurements, 1):
                    st.write(f"{i}. **{m.get('text', 'N/A')}** = {m.get('value', 0)} {m.get('unit', 'FT')} ({m.get('type', 'unknown')})")
            else:
                st.write("No lot measurements found")
        
        with measurement_tabs[1]:
            setback_measurements = analysis_results.get('setback_measurements', [])
            if setback_measurements:
                st.write(f"**Found {len(setback_measurements)} setback measurements:**")
                for i, m in enumerate(setback_measurements, 1):
                    confidence = m.get('confidence', 'medium')
                    st.write(f"{i}. **{m.get('text', 'N/A')}** = {m.get('value', 0)} {m.get('unit', 'FT')} ({m.get('type', 'unknown')}) - Confidence: {confidence}")
            else:
                st.write("No setback measurements found")
        
        with measurement_tabs[2]:
            building_measurements = analysis_results.get('building_measurements', [])
            if building_measurements:
                st.write(f"**Found {len(building_measurements)} building measurements:**")
                for i, m in enumerate(building_measurements, 1):
                    st.write(f"{i}. **{m.get('text', 'N/A')}** = {m.get('value', 0)} {m.get('unit', 'FT')} ({m.get('type', 'unknown')})")
            else:
                st.write("No building measurements found")
        
        # Download Results Section
        st.header("💾 Download Analysis Results")

        # Add verification results display
        if 'verification_results' in st.session_state:
            st.subheader("🔍 Official Verification Results")
        
            verification_data = st.session_state.verification_results.get('verification_results', {})
            official_dims = st.session_state.get('official_dimensions', {})
        
            # Verification summary
            verified_count = sum(1 for v in verification_data.values() if v['status'] == 'VERIFIED')
            total_count = len(verification_data)
        
            if total_count > 0:
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric("Verified Measurements", f"{verified_count}/{total_count}")
                with col_v2:
                    accuracy = (verified_count / total_count) * 100
                    st.metric("Verification Rate", f"{accuracy:.1f}%")
                with col_v3:
                    discrepancies = sum(1 for v in verification_data.values() if v['status'] == 'SIGNIFICANT_DISCREPANCY')
                    st.metric("Discrepancies", discrepancies)
            
                # Detailed verification table
                verification_table_data = []
                for measurement, result in verification_data.items():
                    verification_table_data.append({
                        'Measurement': measurement.replace('_', ' ').title(),
                        'Extracted': result['extracted_value'],
                        'Official': result['official_value'], 
                        'Difference %': f"{result['difference_pct']:.1f}%",
                        'Status': result['status'],
                        'Confidence': result['confidence']
                    })
            
                if verification_table_data:
                    df_verify = pd.DataFrame(verification_table_data)
                
                    def style_verification_status(val):
                        if val == 'VERIFIED':
                            return 'background-color: #d4edda; color: #155724'
                        elif val == 'ACCEPTABLE_VARIANCE':
                            return 'background-color: #fff3cd; color: #856404'
                        else:
                            return 'background-color: #f8d7da; color: #721c24'
                
                    styled_df_verify = df_verify.style.applymap(style_verification_status, subset=['Status'])
                    st.dataframe(styled_df_verify, use_container_width=True, hide_index=True)
        else:
            st.info("No verification data available")

        col_download1, col_download2, col_download3 = st.columns(3)
        
        with col_download1:
            if st.button("📦 Download Complete Package"):
                try:
                    with st.spinner("Creating analysis package..."):
                        zip_data = create_comprehensive_zip(
                            analysis_results,
                            st.session_state.get('extracted_data'),
                            st.session_state.get('pdf_name', 'analysis')
                        )
                        
                        st.download_button(
                            label="📦 Download Analysis Package (ZIP)",
                            data=zip_data,
                            file_name=f"architecture_analysis.zip",
                            mime="application/zip",
                            help="Complete package with measurements, compliance analysis, and extracted regions"
                        )
                except Exception as e:
                    st.error(f"Failed to create download package: {str(e)}")
        
        with col_download2:
            try:
                analysis_json = json.dumps(analysis_results, indent=2, default=str)
                st.download_button(
                    label="📋 Download Analysis (JSON)",
                    data=analysis_json,
                    file_name=f"architecture_analysis.json",
                    mime="application/json",
                    help="Complete analysis results in JSON format"
                )
            except Exception as e:
                st.error(f"Failed to create JSON download: {str(e)}")
        
        with col_download3:
            try:
                if zoning_table:
                    zoning_data = []
                    for req, data in zoning_table.items():
                        if isinstance(data, dict):
                            zoning_data.append({
                                'Requirement': req,
                                'Value': data.get('value', 'N/A'),
                                'Compliance': data.get('compliance', 'UNKNOWN'),
                                'Variance_Needed': data.get('variance_needed', 'N/A')
                            })
                    
                    if zoning_data:
                        df = pd.DataFrame(zoning_data)
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📊 Download Zoning Table (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"zoning_table.csv",
                            mime="text/csv",
                            help="Zoning compliance table in CSV format"
                        )
                    else:
                        st.info("No zoning table data to download")
                else:
                    st.info("No zoning table available for download")
            except Exception as e:
                st.error(f"Failed to create CSV download: {str(e)}")
    
    # Instructions and Help
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        ## Complete Workflow:
        
        ### Step 1: PDF Processing & YOLO Detection
        1. **Upload PDF**: Select an architectural drawing or zoning document (max 25MB for cloud)
        2. **Page Selection**: Choose the page containing the main site plan
        3. **Run YOLO Detection**: AI identifies architectural elements and regions
        
        ### Step 2: Region Extraction & Measurement Analysis
        1. **Extract Regions**: Automatically crops detected architectural elements 
        2. **AI Analysis**: Gemini AI analyzes the best region for:
           - **Lot Measurements**: Area, width, depth, boundary dimensions
           - **Setback Measurements**: Front, rear, side setbacks from dwelling to boundaries
           - **Building Measurements**: Building area, dimensions, height
        
        ### Step 3: Results & Download
        1. **Zoning Analysis**: Complete compliance analysis against R-4 standards
        2. **Interactive Tables**: All measurements with compliance status
        3. **Download Options**: Complete analysis package with all data
        
        ## Cloud Optimizations:
        - Reduced image resolution for faster processing
        - Memory-optimized PDF conversion
        - Shorter API timeouts
        - File size limits to prevent crashes
        
        ## Technology Stack:
        - **YOLOv8**: Object detection for architectural elements
        - **Gemini AI**: Advanced measurement analysis and OCR
        - **PyMuPDF**: Cloud-optimized PDF processing
        - **Streamlit**: Interactive web interface
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Comprehensive Architectural Analysis Pipeline** 🏗️  
    Cloud-Optimized Version • Built with: Streamlit • YOLOv8 • Gemini AI • PyMuPDF  
    Features: PDF Processing • Object Detection • AI Measurement Analysis • Zoning Compliance
    """)

if __name__ == "__main__":
    main()
