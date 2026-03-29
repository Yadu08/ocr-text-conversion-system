# OCR Text Conversion System

An OCR-based system designed to convert handwritten and printed text into digital format using Tesseract OCR and image preprocessing techniques.

---

## Overview

This project processes input images, enhances them using preprocessing techniques, and extracts text using the Tesseract OCR engine. It aims to improve accuracy for both handwritten and printed text.

---

## Features

* Handwritten and printed text recognition
* Image preprocessing for better OCR accuracy
* Automated text extraction pipeline
* Structured digital text output

---

## Tech Stack

* Python
* Tesseract OCR
* OpenCV
* EasyOCR
* CustomTkinter (GUI)

---

## How It Works

1. Input image is provided
2. Image is preprocessed (noise removal, thresholding)
3. OCR engine extracts text
4. Output is generated in readable format

---

## Demo

### Step 1: Upload Image

![Input](images/input.png)

### Step 2: Processing Interface

![Dashboard](images/dashboard.png)

### Step 3: Extracted Output

![Output](images/output.png)

### (Optional) Application Interface

![Login](images/login.png)

---

## Installation & Setup

### 1. Prerequisites

* Python 3.11
  Verify installation:

  ```
  python --version
  ```

---

### 2. Install Tesseract OCR

Download and install Tesseract OCR:

Default path used:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Add Tesseract to system PATH
OR update the path inside the script.

---

### 3. Create Virtual Environment (Recommended)

```
python -m venv venv311
venv311\Scripts\activate
```

---

### 4. Install Dependencies

```
pip install --upgrade pip wheel setuptools
pip install opencv-python easyocr google-generativeai pytesseract pillow reportlab customtkinter torch torchvision scikit-image
```

---

### 5. Configure API Key (Gemini)

Replace API key in the script:

```
GEMINI_API_KEY = "your_api_key_here"
```

Get your API key from:
https://makersuite.google.com/app/apikey

---

### 6. Run the Application

```
python src/ocr_pipeline.py
```

---

### 7. Notes

* Ensure Tesseract is correctly installed and accessible
* EasyOCR downloads models during first run (internet required)
* `customtkinter` requires Tkinter (included with Python)
* SQLite database (`notes.db`) is created automatically
* Output is displayed in the interface and can be copied or exported

---

## Use Cases

* Digitizing handwritten notes
* Extracting text from documents/images
* Academic and document processing

---

## Future Improvements

* Improve accuracy for complex handwriting
* Add GUI enhancements
* Multi-language support

---
