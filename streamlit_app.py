import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
import io
from PIL import Image, ImageFile
import os
from pathlib import Path
import tempfile

# Test page
st.title("Test - Basic Imports Working")
st.write("If you see this, the basic imports are working")
st.write(f"OpenCV version: {cv2.__version__}")
st.write(f"PyMuPDF version: {fitz.__version__}")
