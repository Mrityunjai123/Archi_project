import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes, convert_from_path
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

# Fix for large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuration
GEMINI_KEY = "AIzaSyARUWP7nhktvAnKqS3QgjEVEUKfSl_8iPw"  # Replace with your actual API key
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_KEY}"

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
    """Convert PDF to high-resolution images"""
    
    def __init__(self, dpi=600):
        self.dpi = dpi
    
    def convert_pdf_to_images(self, pdf_file, max_dimension=8000):
        try:
            poppler_path = None
            if os.name == 'nt':  # Windows
                potential_paths = [
                    r"C:\Program Files\poppler-25.07.0\Library\bin",
                    r"C:\Program Files\poppler\bin",
                    r"C:\poppler\bin",
                    r"C:\poppler-25.07.0\Library\bin",
                    r"C:\Program Files (x86)\poppler\bin"
                ]
                for path in potential_paths:
                    if os.path.exists(path) and os.path.exists(os.path.join(path, "pdftoppm.exe")):
                        poppler_path = path
                        break
            
            if isinstance(pdf_file, bytes):
                if poppler_path:
                    images = convert_from_bytes(pdf_file, dpi=self.dpi, poppler_path=poppler_path)
                else:
                    images = convert_from_bytes(pdf_file, dpi=self.dpi)
            else:
                if poppler_path:
                    images = convert_from_path(pdf_file, dpi=self.dpi, poppler_path=poppler_path)
                else:
                    images = convert_from_path(pdf_file, dpi=self.dpi)
            
            processed_images = []
            
            for i, image in enumerate(images):
                original_w, original_h = image.size
                st.info(f"Page {i+1}: Original size {original_w}x{original_h} ({(original_w*original_h)/1_000_000:.1f}M pixels)")
                
                img_array = np.array(image)
                
                h, w = img_array.shape[:2]
                if h > max_dimension or w > max_dimension:
                    if h > w:
                        new_h = max_dimension
                        new_w = int(w * (new_h / h))
                    else:
                        new_w = max_dimension
                        new_h = int(h * (new_w / w))
                    
                    img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                processed_image = Image.fromarray(img_array)
                processed_images.append(processed_image)
                
            return processed_images
            
        except Exception as e:
            st.error(f"Error converting PDF: {str(e)}")
            return []

class YOLOv8Pipeline:
    """YOLOv8 Inference Pipeline with Zone Detection"""
    
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.custom_model = True
                self.classes = list(self.model.names.values()) if hasattr(self.model, 'names') else ["Architecture Layout", "Tables", "zone area"]
            except Exception as e:
                st.warning(f"Could not load custom model: {e}. Using default YOLOv8 model.")
                self.model = YOLO('yolov8n.pt')
                self.custom_model = False
                self.classes = list(self.model.names.values())
        else:
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
        self.analysis_requirements = {
            'min_lot_measurements': 4,  # 4 boundary dimensions + area
            'min_setback_measurements': 1,  # At least 1 valid setback
            'min_building_measurements': 1,  # At least building area
            'required_architecture_count': 2  # Expected number of architectures
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
                if detection_count >= 3:  # Expect multiple regions
                    validation_result['passed'] = True
                    self.required_steps['yolo_completed'] = True
                else:
                    validation_result['issues'].append(f"Only {detection_count} detections found, expected at least 3")
                    validation_result['recommendations'].append("Lower confidence threshold or check image quality")
            else:
                validation_result['issues'].append("YOLO detection failed or no detections found")
                validation_result['retry_needed'] = True
        
        elif step_name == 'regions_extracted':
            if data and data.get('cropped_images') and len(data['cropped_images']) >= 2:
                architecture_regions = [k for k in data['cropped_images'].keys() if 'architecture' in k.lower()]
                if len(architecture_regions) >= 2:
                    validation_result['passed'] = True
                    self.required_steps['regions_extracted'] = True
                    self.required_steps['architectures_identified'] = True
                else:
                    validation_result['issues'].append(f"Only {len(architecture_regions)} architecture regions found, expected 2+")
                    validation_result['recommendations'].append("Review region scoring logic")
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
    
    def force_complete_analysis(self, architectural_regions: List[Dict], detectors: Dict) -> List[Dict]:
        """Reinforcement mechanism to ensure all architectures are analyzed"""
        complete_results = []
        
        st.write("🔄 **REINFORCEMENT LAYER: Ensuring Complete Analysis**")
        
        for i, region_data in enumerate(architectural_regions, 1):
            region_filename = region_data['filename']
            region_image = region_data['image']
            
            st.write(f"🎯 **Forcing Analysis {i}/{len(architectural_regions)}: {region_filename}**")
            
            # Force analysis with retry mechanism
            max_retries = 2
            analysis_success = False
            
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    st.warning(f"Retry attempt {attempt} for {region_filename}")
                
                try:
                    # Force lot analysis
                    st.write(f"📏 Forcing lot analysis (attempt {attempt + 1})...")
                    lot_results = detectors['lot'].detect_lot_measurements(region_image)
                    lot_validation = self.validate_step('measurements_analysis', lot_results if lot_results.get('success') else {'lot_measurements': []})
                    
                    if not lot_validation['passed']:
                        st.error(f"Lot analysis issues: {', '.join(lot_validation['issues'])}")
                    
                    # Force setback analysis
                    st.write(f"📐 Forcing setback analysis (attempt {attempt + 1})...")
                    setback_results = detectors['setback'].detect_setback_measurements(region_image)
                    setback_validation = self.validate_step('measurements_analysis', setback_results if setback_results.get('success') else {'setback_measurements': []})
                    
                    if not setback_validation['passed']:
                        st.error(f"Setback analysis issues: {', '.join(setback_validation['issues'])}")
                    
                    # Force building analysis
                    st.write(f"🏠 Forcing building analysis (attempt {attempt + 1})...")
                    building_results = detectors['building'].detect_building_measurements(region_image)
                    building_validation = self.validate_step('measurements_analysis', building_results if building_results.get('success') else {'building_measurements': []})
                    
                    if not building_validation['passed']:
                        st.error(f"Building analysis issues: {', '.join(building_validation['issues'])}")
                    
                    # Check if analysis is acceptable
                    total_measurements = (
                        len(lot_results.get('measurements', [])) +
                        len(setback_results.get('measurements', [])) +
                        len(building_results.get('measurements', []))
                    )
                    
                    if total_measurements >= 3:  # Minimum acceptable
                        analysis_success = True
                        st.success(f"✅ Analysis {i} completed with {total_measurements} total measurements")
                        break
                    else:
                        st.warning(f"⚠️ Only {total_measurements} measurements found, retrying...")
                
                except Exception as e:
                    st.error(f"❌ Analysis attempt {attempt + 1} failed: {str(e)}")
            
            if not analysis_success:
                st.error(f"❌ Failed to complete analysis for {region_filename} after {max_retries + 1} attempts")
                # Create minimal result to avoid complete failure
                lot_results = {'success': False, 'measurements': [], 'error': 'Analysis failed'}
                setback_results = {'success': False, 'measurements': [], 'error': 'Analysis failed'}
                building_results = {'success': False, 'measurements': [], 'error': 'Analysis failed'}
            
            # Compile results regardless of success level
            yolo_summary = {}  # Placeholder
            analysis_results = detectors['compiler'].compile_zoning_analysis(
                lot_results, setback_results, building_results, 
                region_filename, yolo_summary
            )
            
            # Add reinforcement metadata
            analysis_results['reinforcement_info'] = {
                'analysis_attempts': attempt + 1,
                'forced_completion': True,
                'success_level': 'full' if analysis_success else 'partial',
                'total_measurements_found': (
                    len(analysis_results.get('lot_measurements', [])) +
                    len(analysis_results.get('setback_measurements', [])) +
                    len(analysis_results.get('building_measurements', []))
                )
            }
            
            analysis_results['region_info'] = {
                'filename': region_filename,
                'score': region_data['score'],
                'size': f"{region_image.size[0]}x{region_image.size[1]}",
                'analysis_number': i
            }
            
            complete_results.append(analysis_results)
        
        # Final validation
        st.write("🔍 **FINAL VALIDATION:**")
        if len(complete_results) >= 2:
            st.success(f"✅ Successfully analyzed {len(complete_results)} architectures")
            self.required_steps['all_analyses_completed'] = True
        else:
            st.error(f"❌ Only completed {len(complete_results)} analyses, expected 2+")
        
        return complete_results
    
    def get_completion_report(self) -> str:
        """Generate completion status report"""
        completed_steps = sum(self.required_steps.values())
        total_steps = len(self.required_steps)
        
        report = f"REINFORCEMENT REPORT:\n"
        report += f"Completion: {completed_steps}/{total_steps} steps\n"
        
        for step, completed in self.required_steps.items():
            status = "✅" if completed else "❌"
            report += f"{status} {step.replace('_', ' ').title()}\n"
        
        return report

class EnhancedLotMeasurementsDetector:
    """Enhanced detector for all lot polygon dimensions with label recognition"""
    
    def detect_lot_measurements(self, image_path_or_pil) -> Dict[str, Any]:
        try:
            if isinstance(image_path_or_pil, (str, Path)):
                with open(image_path_or_pil, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                # PIL Image
                img_buffer = io.BytesIO()
                image_path_or_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Failed to encode image: {e}'}
        
        enhanced_lot_prompt = """
ENHANCED LOT POLYGON MEASUREMENTS DETECTION

CRITICAL: Find ALL lot boundary dimensions including labeled measurements with prefixes.

METHOD 1 - EXPLICIT AREA LABELS:
• "AREA = X SF" or "AREA = X SQ FT" 
• "X SQUARE FEET" or "X SQ.FT."
• "X ACRES" (convert: 1 acre = 43,560 SF)

METHOD 2 - LABELED DIMENSION DETECTION (CRITICAL):
• Look for measurements with prefixes: "L=", "B=", "W=", "D="
• "L=77.00'" = Length/Width measurement
• "B=125.0'" = Breadth/Depth measurement  
• "R=300.00'" = Radius for curved boundaries
• These are often lot boundary measurements

METHOD 3 - ALL POLYGON BOUNDARY DIMENSIONS:
• Property boundaries form closed polygon around entire lot
• Each side of polygon should have dimension measurement
• Look for ALL boundary measurements (not just width/depth)
• Include curved boundary measurements (radius, arc length)
• Street boundaries, side boundaries, rear boundaries

Return JSON with ALL found measurements:
[
  {
    "text": "AREA=8,694 S.F.",
    "value": 8694,
    "unit": "SF", 
    "type": "lot_area",
    "method": "explicit_label"
  },
  {
    "text": "L=77.00'",
    "value": 77.0,
    "unit": "FT",
    "type": "lot_width",
    "method": "labeled_dimension"
  }
]

PRIORITY: Find ALL polygon boundary dimensions, especially labeled ones (L=, B=, R=).
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
            response = requests.post(GEMINI_URL, json=payload, timeout=90)
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
                    measurement = {
                        'text': str(item.get('text', '')),
                        'value': float(item.get('value', 0)),
                        'unit': str(item.get('unit', 'FT')),
                        'type': str(item.get('type', 'unknown')),
                        'method': str(item.get('method', 'not_specified'))
                    }
                    if measurement['value'] > 0:
                        measurements.append(measurement)
                
                if measurements:
                    return measurements
        except json.JSONDecodeError:
            pass
        
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
                image_path_or_pil.save(img_buffer, format='PNG')
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
            response = requests.post(GEMINI_URL, json=payload, timeout=90)
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
                    # Strict validation for setbacks
                    if self._validate_setback_measurement(item):
                        measurement = {
                            'text': str(item.get('text', '')),
                            'value': float(item.get('value', 0)),
                            'unit': str(item.get('unit', 'FT')),
                            'type': str(item.get('type', 'unknown')),
                            'measurement_type': str(item.get('measurement_type', 'unspecified')),
                            'confidence': str(item.get('confidence', 'medium'))
                        }
                        measurements.append(measurement)
                
                if measurements:
                    return measurements
        except json.JSONDecodeError:
            pass
        
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
                image_path_or_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return {'success': False, 'error': f'Failed to encode image: {e}'}
        
        enhanced_building_prompt = """
ENHANCED BUILDING MEASUREMENTS DETECTION WITH COMPREHENSIVE POLYGON ANALYSIS

STEP 1: IDENTIFY DWELLING STRUCTURE (CRITICAL)
• Find dwelling labels: "DWELLING", "HOUSE", "RESIDENCE", "BUILDING"
• Look for: "1½ STORY DWELLING", "FRAME DWELLING", "MAIN DWELLING"
• The dwelling is shown as a hatched/shaded polygon (cross-hatched pattern)
• Distinguish from: "GARAGE", "SHED", "PORCH", "ACCESSORY BUILDING"

STEP 2: COMPREHENSIVE BUILDING POLYGON ANALYSIS
• The dwelling polygon represents the building footprint
• Look for ALL dimensions that measure parts of this polygon
• Include main dimensions, wings, extensions, interior spaces
• Building polygon is SEPARATE from lot boundaries

COMPREHENSIVE MEASUREMENTS TO FIND:

1. DWELLING AREA (HIGHEST PRIORITY):
   METHOD A - Explicit Area Labels:
   • "BUILDING AREA = X SF", "FLOOR AREA = X SF", "FOOTPRINT = X SF"
   • "1,152 S.F. FOOTPRINT", "DWELLING AREA: X SF"
   
   METHOD B - Complete Dimensional Analysis:
   • Find ALL building dimensions and calculate total area

2. ALL BUILDING DIMENSIONS (COMPREHENSIVE):
   • Main exterior dimensions (overall width × length)
   • Interior dimensions within building footprint  
   • Wing/extension dimensions
   • Component dimensions (porches, bays, additions)
   • ANY measurement that defines part of the building structure

3. DWELLING HEIGHT INFORMATION:
   • Story information: "1-STORY", "2-STORY", "1½ STORY"
   • Height measurements: "32 FT HEIGHT", "35' MAX HEIGHT"

Return JSON with ALL building dimensions and calculated area:
[
  {
    "text": "26.63' (main building width)",
    "value": 26.63,
    "unit": "FT",
    "type": "dwelling_width",
    "component": "main_structure"
  },
  {
    "text": "1½ STORY",
    "value": 1.5,
    "unit": "stories",
    "type": "dwelling_height",
    "component": "building_height"
  }
]

MANDATORY: Find ALL building dimensions and calculate accurate total area.
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
            response = requests.post(GEMINI_URL, json=payload, timeout=90)
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
                    measurement = {
                        'text': str(item.get('text', '')),
                        'value': float(item.get('value', 0)),
                        'unit': str(item.get('unit', 'FT')),
                        'type': str(item.get('type', 'unknown')),
                        'component': str(item.get('component', 'unspecified'))
                    }
                    if measurement['value'] > 0 and 'dwelling' in measurement['type']:
                        measurements.append(measurement)
                
                if measurements:
                    return measurements
        except json.JSONDecodeError:
            pass
        
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
    
    def _process_setback_measurements(self, measurements):
        summary = {}
        
        setback_types = {
            'front_setback': 'front',
            'rear_setback': 'rear',
            'left_side_setback': 'left_side',
            'right_side_setback': 'right_side'
        }
        
        for m_type, key in setback_types.items():
            setbacks = [m for m in measurements if m['type'] == m_type]
            if setbacks:
                summary[key] = setbacks[0]
        
        # Calculate minimum side and aggregate
        sides = [summary.get('left_side'), summary.get('right_side')]
        sides = [s for s in sides if s]
        
        if sides:
            summary['min_side'] = min(sides, key=lambda x: x['value'])
            if len(sides) == 2:
                summary['aggregate'] = {
                    'value': sum(s['value'] for s in sides),
                    'unit': 'FT',
                    'text': f"Aggregate: {sides[0]['value']:.2f}' + {sides[1]['value']:.2f}'"
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
    
    st.header("📊 Comprehensive Zoning Analysis Dashboard")
    
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
    st.subheader("🏗️ Zoning Requirements Analysis")
    
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
    tabs = st.tabs(["📏 Lot Measurements", "📐 Setback Measurements", "🏠 Building Measurements"])
    
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
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model path - Updated to use user's specific model
    model_path = r"D:\custom_yolov8_final.pt"
    
    if os.path.exists(model_path):
        st.sidebar.success(f"✅ Using custom YOLO model: {os.path.basename(model_path)}")
        st.sidebar.info("Model supports zone detection and architectural elements")
    else:
        st.sidebar.error(f"❌ Custom model not found: {model_path}")
        st.sidebar.warning("Please verify the model path is correct")
    
    # Configuration parameters
    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.25, step=0.05)
    
    # Gemini API key check
    if GEMINI_KEY:
        st.sidebar.success("✅ Gemini AI configured for measurement analysis")
    else:
        st.sidebar.error("❌ Gemini API key not configured")
        st.sidebar.warning("Please set your Gemini API key in the code")
    
    # Zoning information
    st.sidebar.markdown("**Zoning Support:**")
    zones_supported = list(LIVINGSTON_ZONING_REQUIREMENTS.keys())
    st.sidebar.info(f"Supports {len(zones_supported)} zones: {', '.join(zones_supported[:5])}{'...' if len(zones_supported) > 5 else ''}")
    
    # Fixed settings display
    st.sidebar.markdown("**Processing Settings:**")
    st.sidebar.info("PDF DPI: 600 (High Quality)")
    st.sidebar.info("Max Image Dimension: 8000px")
    st.sidebar.info("Zoning Data: Livingston Township")
    
    # Initialize components with reinforcement layer
    pdf_converter = PDFtoImageConverter(dpi=600)
    yolo_pipeline = YOLOv8Pipeline(model_path=model_path)
    validator = AnalysisValidator()
    lot_detector = EnhancedLotMeasurementsDetector()
    setback_detector = EnhancedSetbackMeasurementsDetector()
    building_detector = EnhancedBuildingMeasurementsDetector()
    zoning_compiler = ZoningTableCompiler()
    
    # Create detector dictionary for reinforcement layer
    detector_dict = {
        'lot': lot_detector,
        'setback': setback_detector, 
        'building': building_detector,
        'compiler': zoning_compiler
    }
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload")
        uploaded_pdf = st.file_uploader(
            "Upload PDF file with architectural drawings",
            type=['pdf'],
            help="Upload a PDF containing architectural plans or zoning documents"
        )
        
        if uploaded_pdf:
            st.success(f"PDF uploaded: {uploaded_pdf.name}")
            
            # Convert PDF to images
            with st.spinner("Converting PDF to high-resolution images..."):
                pdf_bytes = uploaded_pdf.getvalue()
                images = pdf_converter.convert_pdf_to_images(pdf_bytes, max_dimension=8000)
            
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
                
                # Step 1: YOLO Detection with Reinforcement
                if st.button("🔍 Step 1: Run YOLO Detection", type="primary"):
                    with col2:
                        with st.spinner("Running YOLO detection with reinforcement layer..."):
                            annotated_image, results = yolo_pipeline.predict_image(selected_image, conf_threshold=conf_threshold)
                            
                            # Validate YOLO results
                            yolo_validation = validator.validate_step('yolo_completed', results)
                            
                            if not yolo_validation['passed']:
                                st.error("🔄 YOLO Detection Issues:")
                                for issue in yolo_validation['issues']:
                                    st.error(f"• {issue}")
                                for rec in yolo_validation['recommendations']:
                                    st.info(f"💡 {rec}")
                                
                                if yolo_validation['retry_needed']:
                                    st.warning("Retrying with lower confidence threshold...")
                                    annotated_image, results = yolo_pipeline.predict_image(selected_image, conf_threshold=0.1)
                                    
                                    # Re-validate
                                    yolo_validation = validator.validate_step('yolo_completed', results)
                                    if yolo_validation['passed']:
                                        st.success("✅ Retry successful!")
                                    else:
                                        st.error("❌ Retry failed - proceeding with available detections")
                            
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
                                st.success("✅ YOLO detection passed validation")
                            else:
                                st.warning("⚠️ YOLO detection has validation issues but proceeding")
    
    # Step 2: Extract and Analyze (only show if YOLO detection is complete)
    if ('detection_results' in st.session_state and 
        st.session_state.detection_results is not None):
        
        st.markdown("---")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.header("🔬 Step 2: Extract & Analyze Measurements")
            
            if st.button("🚀 Extract Regions & Analyze Measurements", type="primary"):
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
                        # Improved region selection for architectural analysis
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
                                score += 1000  # Highest priority for architectural drawings
                                st.write(f"  ✅ Architecture/Layout detected (+1000)")
                            elif 'site' in filename.lower() or 'plan' in filename.lower():
                                score += 800   # High priority for site plans
                                st.write(f"  ✅ Site/Plan detected (+800)")
                            elif 'zone' in filename.lower() and 'map' not in filename.lower():
                                score += 600   # Medium-high priority for zone drawings
                                st.write(f"  ✅ Zone drawing detected (+600)")
                            elif 'table' in filename.lower() or 'calculation' in filename.lower():
                                score -= 500   # Strongly deprioritize tables and calculations
                                st.write(f"  ❌ Table/Calculation detected (-500)")
                            
                            # Size bonus for larger regions (likely to be main drawings)
                            if area > 200000:  # Large region
                                score += 200
                                st.write(f"  ✅ Large region (+200)")
                            elif area > 100000:  # Medium region
                                score += 100
                                st.write(f"  ✅ Medium region (+100)")
                            
                            # Aspect ratio bonus for typical architectural drawings
                            aspect_ratio = max(width, height) / min(width, height)
                            if 1.2 <= aspect_ratio <= 2.0:  # Typical architectural drawing proportions
                                score += 50
                                st.write(f"  ✅ Good aspect ratio (+50)")
                            
                            st.write(f"  **Final Score: {score}**")
                            
                            if score > best_score:
                                best_score = score
                                best_region = image
                                best_filename = filename
                        
                        # If no clearly architectural region found, pick the largest non-table region
                        if best_score <= 0:
                            st.warning("No architectural drawings detected. Selecting largest non-table region...")
                            non_table_regions = [(f, img) for f, img in extracted_data['cropped_images'].items() 
                                                if not any(word in f.lower() for word in ['table', 'calculation', 'soil'])]
                            
                            if non_table_regions:
                                best_filename, best_region = max(
                                    non_table_regions,
                                    key=lambda x: x[1].size[0] * x[1].size[1]
                                )
                            else:
                                # Last resort: use any region except tables
                                filtered_regions = [(f, img) for f, img in extracted_data['cropped_images'].items() 
                                                  if 'table' not in f.lower()]
                                if filtered_regions:
                                    best_filename, best_region = max(
                                        filtered_regions,
                                        key=lambda x: x[1].size[0] * x[1].size[1]
                                    )
                                else:
                                    best_filename, best_region = max(
                                        extracted_data['cropped_images'].items(),
                                        key=lambda x: x[1].size[0] * x[1].size[1]
                                    )
                        
                        if best_region:
                            st.success(f"🎯 **Selected for analysis: {best_filename}** (Score: {best_score})")
                            st.info("This region should contain lot boundaries, building footprint, and dimensional measurements.")
                        
                        if best_region:
                            st.info(f"Analyzing measurements from region: {best_filename}")
                            
                            # Show the region being analyzed for debugging
                            st.subheader("Region Being Analyzed:")
                            st.image(best_region, caption=f"Analyzing: {best_filename}", use_column_width=True)
                            
                            # Run all three measurement analyses with detailed logging
                            st.write("🔍 **Running Lot Measurements Analysis...**")
                            lot_results = lot_detector.detect_lot_measurements(best_region)
                            if lot_results.get('success'):
                                st.success(f"✅ Lot analysis: Found {len(lot_results.get('measurements', []))} measurements")
                            else:
                                st.error(f"❌ Lot analysis failed: {lot_results.get('error', 'Unknown error')}")
                            
                            st.write("🔍 **Running Setback Measurements Analysis...**")
                            setback_results = setback_detector.detect_setback_measurements(best_region)
                            if setback_results.get('success'):
                                st.success(f"✅ Setback analysis: Found {len(setback_results.get('measurements', []))} measurements")
                            else:
                                st.error(f"❌ Setback analysis failed: {setback_results.get('error', 'Unknown error')}")
                            
                            st.write("🔍 **Running Building Measurements Analysis...**")
                            building_results = building_detector.detect_building_measurements(best_region)
                            if building_results.get('success'):
                                st.success(f"✅ Building analysis: Found {len(building_results.get('measurements', []))} measurements")
                            else:
                                st.error(f"❌ Building analysis failed: {building_results.get('error', 'Unknown error')}")
                            
                            # Show raw API responses for debugging
                            if st.checkbox("Show Raw API Responses (Debug)", key="show_debug"):
                                st.subheader("🔧 Debug Information")
                                
                                if lot_results.get('raw_response'):
                                    with st.expander("Lot Analysis Raw Response"):
                                        st.text(lot_results['raw_response'][:1000] + "..." if len(lot_results['raw_response']) > 1000 else lot_results['raw_response'])
                                
                                if setback_results.get('raw_response'):
                                    with st.expander("Setback Analysis Raw Response"):
                                        st.text(setback_results['raw_response'][:1000] + "..." if len(setback_results['raw_response']) > 1000 else setback_results['raw_response'])
                                
                                if building_results.get('raw_response'):
                                    with st.expander("Building Analysis Raw Response"):
                                        st.text(building_results['raw_response'][:1000] + "..." if len(building_results['raw_response']) > 1000 else building_results['raw_response'])
                            
                            # Get YOLO detection summary for zone detection
                            yolo_summary = yolo_pipeline.get_detection_summary(st.session_state.detection_results)
                            
                            # Compile comprehensive analysis with zone detection
                            analysis_results = zoning_compiler.compile_zoning_analysis(
                                lot_results, setback_results, building_results, 
                                best_filename, yolo_summary
                            )
                            
                            st.session_state.analysis_results = analysis_results
                            
                            # Display results in the second column
                            with col4:
                                st.header("📊 Measurement Analysis Results")
                                
                                # Show detected zone information
                                detected_zone = analysis_results['summary'].get('detected_zone', 'Unknown')
                                zone_use = analysis_results['summary'].get('zone_use', 'Unknown')
                                
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
                                    st.error("❌ No measurement analyses completed successfully")
                        else:
                            st.warning("No suitable architectural regions found for measurement analysis")
                else:
                    st.warning("No regions extracted from YOLO detection")
    
    # Step 3: Multiple Architecture Analysis Dashboard (show if any analysis exists)
    if ('multiple_analysis_results' in st.session_state and st.session_state.multiple_analysis_results) or ('analysis_results' in st.session_state and st.session_state.analysis_results):
        st.markdown("---")
        
        # Use multiple results if available, otherwise single result
        analysis_list = st.session_state.get('multiple_analysis_results', [])
        if not analysis_list and 'analysis_results' in st.session_state and st.session_state.analysis_results:
            analysis_list = [st.session_state.analysis_results]
        
        if not analysis_list:
            st.error("No analysis results found")
            return
            
        # Display separate analysis for each architectural layout
        st.header("📊 Step 3: Comprehensive Architecture Analysis Results")
        
        st.info(f"Found {len(analysis_list)} architectural layout(s) for detailed analysis")
        
        for i, analysis_results in enumerate(analysis_list, 1):
            region_info = analysis_results.get('region_info', {})
            region_filename = region_info.get('filename', f'Architecture {i}')
            
            st.subheader(f"🏗️ Architecture Analysis {i}: {region_filename}")
            
            # Show the architectural drawing being analyzed
            col_arch1, col_arch2 = st.columns([1, 2])
            
            with col_arch1:
                st.markdown("**Architectural Drawing:**")
                # Find and display the corresponding image
                if 'extracted_data' in st.session_state:
                    extracted_images = st.session_state.extracted_data.get('cropped_images', {})
                    if region_filename in extracted_images:
                        st.image(extracted_images[region_filename], 
                                caption=f"Source: {region_filename}", 
                                use_column_width=True)
                    else:
                        st.info("Image not available for this analysis")
                
                # Show detected zone and summary
                detected_zone = analysis_results.get('summary', {}).get('detected_zone', 'R-4')
                zone_use = analysis_results.get('summary', {}).get('zone_use', 'Single Family')
                
                st.metric("Detected Zone", detected_zone)
                st.metric("Zone Use", zone_use)
            
            with col_arch2:
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
            
            # Individual Zoning Requirements Table for this Architecture
            st.markdown(f"**Zoning Requirements Table - Architecture {i}:**")
            
            zoning_table = analysis_results.get('zoning_table', {})
            if zoning_table:
                # Create DataFrame for this architecture's analysis
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
                
                if table_data:  # Only show table if we have data
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
                    st.warning("No zoning table data available for this architecture")
            else:
                st.warning("No zoning table found for this architecture")
            
            # Detailed measurements for this architecture
            measurement_tabs = st.tabs([f"📏 Lot Measurements", f"📐 Setback Measurements", f"🏠 Building Measurements"])
            
            with measurement_tabs[0]:
                lot_measurements = analysis_results.get('lot_measurements', [])
                if lot_measurements:
                    st.write(f"**Found {len(lot_measurements)} lot measurements:**")
                    for j, m in enumerate(lot_measurements, 1):
                        st.write(f"{j}. **{m.get('text', 'N/A')}** = {m.get('value', 0)} {m.get('unit', 'FT')} ({m.get('type', 'unknown')})")
                else:
                    st.write("No lot measurements found for this architecture")
            
            with measurement_tabs[1]:
                setback_measurements = analysis_results.get('setback_measurements', [])
                if setback_measurements:
                    st.write(f"**Found {len(setback_measurements)} setback measurements:**")
                    for j, m in enumerate(setback_measurements, 1):
                        orientation_method = m.get('orientation_method', 'unknown method')
                        confidence = m.get('confidence', 'medium')
                        st.write(f"{j}. **{m.get('text', 'N/A')}** = {m.get('value', 0)} {m.get('unit', 'FT')} ({m.get('type', 'unknown')}) - {orientation_method} (Confidence: {confidence})")
                else:
                    st.write("No setback measurements found for this architecture")
            
            with measurement_tabs[2]:
                building_measurements = analysis_results.get('building_measurements', [])
                if building_measurements:
                    st.write(f"**Found {len(building_measurements)} building measurements:**")
                    for j, m in enumerate(building_measurements, 1):
                        st.write(f"{j}. **{m.get('text', 'N/A')}** = {m.get('value', 0)} {m.get('unit', 'FT')} ({m.get('type', 'unknown')})")
                else:
                    st.write("No building measurements found for this architecture")
            
            # Individual download section for this architecture
            st.markdown(f"**Download Results for Architecture {i}:**")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download individual analysis JSON
                individual_json = json.dumps(analysis_results, indent=2, default=str)
                st.download_button(
                    label=f"📋 Download Analysis {i} (JSON)",
                    data=individual_json,
                    file_name=f"architecture_{i}_analysis.json",
                    mime="application/json",
                    key=f"download_arch_{i}_json"
                )
            
            with col_dl2:
                # Download individual zoning table CSV
                if zoning_table:
                    individual_csv_data = []
                    for req, data in zoning_table.items():
                        individual_csv_data.append({
                            'Architecture': f'Architecture {i}',
                            'Requirement': req,
                            'Value': data.get('value', 'N/A'),
                            'Compliance': data.get('compliance', 'UNKNOWN'),
                            'Variance_Needed': data.get('variance_needed', 'N/A')
                        })
                    
                    df_individual = pd.DataFrame(individual_csv_data)
                    csv_buffer = io.StringIO()
                    df_individual.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label=f"📊 Download Table {i} (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"architecture_{i}_zoning_table.csv",
                        mime="text/csv",
                        key=f"download_arch_{i}_csv"
                    )
            
            st.markdown("---")
        
        # Combined download for multiple architectures
        if len(analysis_list) > 1:
            st.header("💾 Download Complete Multi-Architecture Analysis")
            
            col_combined1, col_combined2, col_combined3 = st.columns(3)
            
            with col_combined1:
                # Download comprehensive analysis package
                if st.button("📦 Download All Architectures Package"):
                    with st.spinner("Creating comprehensive multi-architecture package..."):
                        zip_data = create_comprehensive_zip(
                            {'multiple_architectures': analysis_list},
                            st.session_state.get('extracted_data'),
                            st.session_state.get('pdf_name', 'analysis')
                        )
                        
                        st.download_button(
                            label="📦 Download Multi-Architecture Package (ZIP)",
                            data=zip_data,
                            file_name=f"multi_architecture_analysis.zip",
                            mime="application/zip",
                            help="Complete package with all architectures, measurements, and compliance analysis"
                        )
            
            with col_combined2:
                # Download combined analysis JSON
                combined_json = json.dumps({
                    'multiple_architectures': analysis_list,
                    'pdf_source': st.session_state.get('pdf_name', 'unknown'),
                    'analysis_count': len(analysis_list)
                }, indent=2, default=str)
                st.download_button(
                    label="📋 Download Combined Analysis (JSON)",
                    data=combined_json,
                    file_name=f"combined_architectures.json",
                    mime="application/json"
                )
            
            with col_combined3:
                # Download combined CSV with all architectures
                combined_csv_data = []
                for i, analysis in enumerate(analysis_list, 1):
                    zoning_table = analysis.get('zoning_table', {})
                    region_info = analysis.get('region_info', {})
                    region_filename = region_info.get('filename', f'Architecture {i}')
                    
                    for req, data in zoning_table.items():
                        combined_csv_data.append({
                            'Architecture_Number': i,
                            'Architecture_Name': region_filename,
                            'Requirement': req,
                            'Value': data.get('value', 'N/A'),
                            'Compliance': data.get('compliance', 'UNKNOWN'),
                            'Variance_Needed': data.get('variance_needed', 'N/A')
                        })
                
                if combined_csv_data:
                    df_combined = pd.DataFrame(combined_csv_data)
                    csv_combined_buffer = io.StringIO()
                    df_combined.to_csv(csv_combined_buffer, index=False)
                    
                    st.download_button(
                        label="📊 Download All Tables (CSV)",
                        data=csv_combined_buffer.getvalue(),
                        file_name=f"all_architectures_zoning.csv",
                        mime="text/csv"
                    )
        else:
            # Single architecture download options
            st.header("💾 Download Analysis Results")
            
            col_single1, col_single2, col_single3 = st.columns(3)
            
            with col_single1:
                if st.button("📦 Download Complete Package"):
                    with st.spinner("Creating analysis package..."):
                        zip_data = create_comprehensive_zip(
                            analysis_list[0],
                            st.session_state.get('extracted_data'),
                            st.session_state.get('pdf_name', 'analysis')
                        )
                        
                        st.download_button(
                            label="📦 Download Analysis Package (ZIP)",
                            data=zip_data,
                            file_name=f"architecture_analysis.zip",
                            mime="application/zip"
                        )
            
            with col_single2:
                analysis_json = json.dumps(analysis_list[0], indent=2, default=str)
                st.download_button(
                    label="📋 Download Analysis (JSON)",
                    data=analysis_json,
                    file_name=f"architecture_analysis.json",
                    mime="application/json"
                )
            
            with col_single3:
                zoning_table = analysis_list[0].get('zoning_table', {})
                if zoning_table:
                    zoning_data = []
                    for req, data in zoning_table.items():
                        zoning_data.append({
                            'Requirement': req,
                            'Value': data.get('value', 'N/A'),
                            'Compliance': data.get('compliance', 'UNKNOWN'),
                            'Variance_Needed': data.get('variance_needed', 'N/A')
                        })
                    
                    df = pd.DataFrame(zoning_data)
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📊 Download Zoning Table (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"zoning_table.csv",
                        mime="text/csv"
                    )
    
    # Instructions
    with st.expander("ℹ️ How to use this comprehensive analysis tool"):
        st.markdown("""
        ## Complete Workflow:
        
        ### Step 1: PDF Processing & YOLO Detection
        1. **Upload PDF**: Select an architectural drawing or zoning document
        2. **Page Selection**: Choose the page containing the main site plan
        3. **Run YOLO Detection**: AI identifies architectural elements, tables, and zones
        
        ### Step 2: Region Extraction & Measurement Analysis
        1. **Extract Regions**: Crops detected architectural elements 
        2. **Gemini AI Analysis**: Advanced AI analyzes each region for:
           - **Lot Measurements**: Area, width, depth, boundary dimensions
           - **Setback Measurements**: Front, rear, side setbacks from dwelling to boundaries
           - **Building Measurements**: Building area, dimensions, height
        
        ### Step 3: Comprehensive Results
        1. **Zoning Analysis Dashboard**: Complete compliance analysis
        2. **Interactive Tables**: All measurements with compliance status
        3. **Download Options**: Complete analysis package with all data
        
        ## What You Get:
        - ✅ Complete zoning requirements analysis
        - ✅ Compliance checking against R-4 zoning standards
        - ✅ Detailed measurement extraction with confidence scores
        - ✅ Comprehensive reports and data exports
        - ✅ Visual annotations and region extractions
        
        ## Technology Stack:
        - **YOLOv8**: Object detection for architectural elements
        - **Gemini AI**: Advanced measurement analysis and OCR
        - **Computer Vision**: High-resolution PDF processing
        - **Compliance Engine**: Automatic zoning compliance checking
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Comprehensive Architectural Analysis Pipeline** 🏗️  
    Built with: Streamlit • YOLOv8 • Gemini AI • OpenCV • Computer Vision  
    Features: PDF Processing • Object Detection • AI Measurement Analysis • Zoning Compliance
    """)

if __name__ == "__main__":
    main()
