import sqlite3
import hashlib
import secrets
import os
import math
import cv2
import easyocr
import google.generativeai as genai
import time
import queue
import concurrent.futures

from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageTk
import pytesseract
from pytesseract import Output
import threading
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Load EasyOCR model (once, globally)
reader = easyocr.Reader(['en'], gpu=False)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDDL5A9t91zbnd3qmurj77pSlO_Xp-MINo"  # Get from https://makersuite.google.com/app/apikey

# Initialize API configuration
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE" and len(GEMINI_API_KEY.strip()) > 0:
        genai.configure(api_key=GEMINI_API_KEY)
        print("[OK] Gemini API configured")
    else:
        print("[WARNING] Gemini API key not configured")
except Exception as e:
    print(f"[WARNING] Error configuring Gemini API: {e}")

# ---------------------- DATABASE LAYER ---------------------- #
DB_NAME = "notes.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            recognized_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            ocr_method TEXT DEFAULT 'easyocr',
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
    """)
    
    # Migration: Add ocr_method column if it doesn't exist (for existing databases)
    try:
        cur.execute("ALTER TABLE notes ADD COLUMN ocr_method TEXT DEFAULT 'easyocr'")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    conn.commit()
    conn.close()

# ---------------------- AUTH / SECURITY ---------------------- #
HASH_ALGO = "sha256"
ITERATIONS = 100_000

def hash_password(password: str) -> str:
    """Return salt$hash using PBKDF2."""
    salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        HASH_ALGO,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        ITERATIONS,
    )
    return f"{salt}${pw_hash.hex()}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, stored_hash = stored.split("$")
    except ValueError:
        return False
    pw_hash = hashlib.pbkdf2_hmac(
        HASH_ALGO,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        ITERATIONS,
    )
    return pw_hash.hex() == stored_hash

def register_user(username: str, email: str, password: str):
    if len(username.strip()) == 0 or len(email.strip()) == 0:
        return False, "Username and email cannot be empty."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    conn = get_connection()
    cur = conn.cursor()
    try:
        pwd_hash = hash_password(password)
        cur.execute("""
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        """, (username, email, pwd_hash, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        conn.close()

def login_user(username: str, password: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, password_hash FROM users WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return False, "Invalid username or password.", None
    user_id, stored_hash = row
    if not verify_password(password, stored_hash):
        return False, "Invalid username or password.", None
    return True, "Login successful.", user_id

# ---------------------- AI STATUS CHECKING ---------------------- #

# Model cache to avoid repeated testing
_model_cache = {
    'working_model': None,
    'working_model_name': None,
    'failed_models': set(),  # Track models that don't work
    'last_check': 0,
    'quota_error_count': 0  # Track quota errors
}

def clear_model_cache():
    """Clear the model cache - useful when quota might have reset."""
    global _model_cache
    _model_cache = {
        'working_model': None,
        'working_model_name': None,
        'failed_models': set(),
        'last_check': 0,
        'quota_error_count': 0
    }

def _should_skip_model(model_name: str) -> bool:
    """Check if a model should be skipped (doesn't support images or is TTS-only)."""
    model_lower = model_name.lower()
    
    # Skip Gemma models (text-only, no image support)
    if 'gemma' in model_lower:
        return True
    
    # Skip TTS models (text-to-speech only)
    if 'tts' in model_lower:
        return True
    
    # Skip robotics models (not for OCR)
    if 'robotics' in model_lower:
        return True
    
    # Skip computer-use models (not for OCR)
    if 'computer-use' in model_lower:
        return True
    
    # Skip deep-research models (not for OCR)
    if 'deep-research' in model_lower:
        return True
    
    # Skip image-generation models (generation only, not OCR)
    if 'image-generation' in model_lower or 'image-preview' in model_lower:
        return True
    
    # Skip nano-banana (not for OCR)
    if 'nano-banana' in model_lower:
        return True
    
    return False

def _check_model_supports_images(model_info) -> bool:
    """Check if a model supports image input."""
    # Check supported input modalities
    if hasattr(model_info, 'input_token_limit'):
        # Models with input_token_limit typically support images
        return True
    
    # Check supported generation methods
    if hasattr(model_info, 'supported_generation_methods'):
        methods = model_info.supported_generation_methods
        # Models that support generateContent with images
        if 'generateContent' in methods:
            return True
    
    return False

def get_available_gemini_model(force_refresh: bool = False):
    """Get the best available Gemini model for vision/OCR with caching."""
    global _model_cache
    
    # Return cached model if available and not forcing refresh
    if not force_refresh and _model_cache['working_model'] is not None:
        current_time = time.time()
        # Cache valid for 5 minutes
        if current_time - _model_cache['last_check'] < 300:
            return _model_cache['working_model_name'], _model_cache['working_model']
    
    # Try to list available models first
    try:
        all_models = list(genai.list_models())
        # Filter for models that support image input
        available_models = []
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods:
                # Check if model supports images by checking input modalities
                if hasattr(m, 'input_token_limit') or _check_model_supports_images(m):
                    model_name = m.name
                    # Skip models that don't support images
                    if not _should_skip_model(model_name):
                        available_models.append(model_name)
        
        if available_models:
            print(f"[INFO] Found {len(available_models)} image-capable models")
    except Exception as e:
        print(f"[DEBUG] Could not list models: {e}")
        available_models = []
    
    # Priority list of models that support images (in order of preference)
    priority_models = [
        'models/gemini-2.5-flash',
        'models/gemini-2.5-pro',
        'models/gemini-2.0-flash-exp',
        'models/gemini-2.0-flash',
        'models/gemini-1.5-pro',
        'models/gemini-1.5-flash',
        'models/gemini-pro-vision',
        'models/gemini-pro',
        'gemini-pro-vision',
        'gemini-pro',
    ]
    
    # Combine priority models with available models, removing duplicates and skipped models
    models_to_try = []
    seen = set()
    
    # Add priority models first
    for model in priority_models:
        if model not in seen and not _should_skip_model(model):
            models_to_try.append(model)
            seen.add(model)
    
    # Add other available models
    for model in available_models:
        if model not in seen and not _should_skip_model(model):
            models_to_try.append(model)
            seen.add(model)
    
    # Remove models that we know don't work
    models_to_try = [m for m in models_to_try if m not in _model_cache['failed_models']]
    
    if not models_to_try:
        print("[ERROR] No suitable Gemini models available")
        return None, None
    
    # Try each model
    for model_name in models_to_try:
        # Skip if we know this model failed before
        if model_name in _model_cache['failed_models']:
            continue
            
        try:
            model = genai.GenerativeModel(model_name)
            # Quick test to verify it works (with timeout to avoid hanging)
            test_response = model.generate_content("test", request_options={"timeout": 5})
            if test_response and hasattr(test_response, 'text'):
                # Cache the working model
                _model_cache['working_model'] = model
                _model_cache['working_model_name'] = model_name
                _model_cache['last_check'] = time.time()
                _model_cache['quota_error_count'] = 0  # Reset quota error count on success
                print(f"[INFO] Using model: {model_name}")
                return model_name, model
        except Exception as e:
            error_str = str(e).lower()
            # Track failed models
            _model_cache['failed_models'].add(model_name)
            
            # Track quota errors
            if "quota" in error_str or "429" in error_str:
                _model_cache['quota_error_count'] = _model_cache.get('quota_error_count', 0) + 1
                # If we get too many quota errors, clear cache and stop trying
                if _model_cache['quota_error_count'] > 3:
                    _model_cache['working_model'] = None
                    _model_cache['working_model_name'] = None
                    continue
            else:
                # Only log if it's not a quota/quota-related error (to reduce noise)
                if "not found" in error_str or "404" in error_str:
                    continue  # Don't log "not found" errors
                elif "image input" in error_str or "modality" in error_str:
                    continue  # Don't log image modality errors (expected for some models)
                else:
                    # Only log unexpected errors
                    pass
            continue
    
    print("[ERROR] No working Gemini models available")
    return None, None

def check_gemini_api_status() -> tuple[bool, str]:
    """Check if Gemini API is configured and working."""
    # Check if API key is configured
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE" or len(GEMINI_API_KEY.strip()) == 0:
        return False, "API key not configured"
    
    # Ensure API is configured
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        return False, f"Configuration error: {str(e)[:50]}"
    
    try:
        # Check if we have a cached working model first
        if _model_cache['working_model'] is not None:
            current_time = time.time()
            # If cache is still valid (less than 5 minutes old), use it
            if current_time - _model_cache['last_check'] < 300:
                return True, f"Ready ({_model_cache['working_model_name']})"
        
        # Try to find an available model with a quick test
        model_name, model = get_available_gemini_model(force_refresh=False)
        if not model:
            # Check if all models failed due to quota
            if len(_model_cache['failed_models']) > 5:
                # If many models failed, likely quota issue
                return False, "Quota exceeded - All models unavailable"
            return False, "No compatible model found"
        
        # Test the model with a very simple request to check quota
        try:
            test_response = model.generate_content("hi", request_options={"timeout": 5})
            if test_response and hasattr(test_response, 'text'):
                return True, f"Ready ({model_name})"
            else:
                return False, "Model test failed"
        except Exception as test_error:
            error_str = str(test_error).lower()
            if "quota" in error_str or "429" in error_str:
                # Clear cache on quota error
                _model_cache['working_model'] = None
                _model_cache['working_model_name'] = None
                return False, "Quota exceeded - Please try again later"
            elif "timeout" in error_str:
                return False, "Connection timeout"
            else:
                return False, f"Test failed: {str(test_error)[:50]}"
        
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "authentication" in error_msg or "invalid" in error_msg:
            return False, "Invalid API key"
        elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
            return False, "Quota exceeded - Please try again later"
        elif "permission" in error_msg or "forbidden" in error_msg:
            return False, "Permission denied"
        elif "timeout" in error_msg or "connection" in error_msg:
            return False, "Connection timeout"
        else:
            return False, f"Error: {str(e)[:60]}"

def get_ocr_engine_status() -> dict:
    """Get status of all OCR engines."""
    status = {}
    
    # Check Gemini
    gemini_ok, gemini_msg = check_gemini_api_status()
    status['gemini'] = {'available': gemini_ok, 'message': gemini_msg}
    
    # Check EasyOCR
    try:
        if reader:
            status['easyocr'] = {'available': True, 'message': 'Ready'}
        else:
            status['easyocr'] = {'available': False, 'message': 'Not loaded'}
    except:
        status['easyocr'] = {'available': False, 'message': 'Error'}
    
    # Check Tesseract
    try:
        pytesseract.get_tesseract_version()
        status['tesseract'] = {'available': True, 'message': 'Ready'}
    except:
        status['tesseract'] = {'available': False, 'message': 'Not installed'}
    
    return status

# ---------------------- CONFIDENCE HELPERS ---------------------- #

def estimate_confidences_from_text(text: str, default_conf: float = 0.65):
    """Heuristic confidence assignment when engine does not return scores."""
    confidences = []
    if not text:
        return confidences
    low_chars = set("?%$#@[]{}()")
    for token in text.split():
        score = default_conf
        if len(token) > 8:
            score += 0.08
        if any(ch.isdigit() for ch in token):
            score -= 0.05
        if any(ch in low_chars for ch in token):
            score -= 0.1
        score = max(0.3, min(0.95, score))
        confidences.append((token, score))
    return confidences

# ---------------------- AI-POWERED OCR LAYER ---------------------- #

def preprocess_image_for_ocr(image_path: str) -> str:
    """Advanced preprocessing optimized for handwritten notebooks."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gentle denoise to preserve handwriting strokes
        gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)

        # Contrast enhancement (CLAHE) tuned for handwriting
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Binarize with adaptive threshold tuned for notebooks
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8
        )

        # Deskew using Hough transform (critical for notebooks)
        edges = cv2.Canny(th, 50, 150)
        lines = cv2.HoughLines(edges, 1, math.pi/180, 120)
        angle = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:,0]:
                deg = (theta * 180.0 / 3.14159265359)
                if 80 < deg < 100 or deg < 10 or deg > 170:
                    continue
                angles.append(deg - 90)
            if angles:
                angle = sum(angles) / len(angles)
        
        if abs(angle) > 0.3:
            (h, w) = th.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            th = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Morphology to strengthen strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Ensure black text on white background
        if th.mean() < 127:
            th = 255 - th

        # Upscale for OCR
        scale = 3
        th = cv2.resize(
            th,
            (th.shape[1] * scale, th.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )

        # Save preprocessed image temporarily
        temp_path = image_path.replace(os.path.splitext(image_path)[1], "_preprocessed.png")
        cv2.imwrite(temp_path, th)
        
        return temp_path

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return image_path


def run_gemini_ocr(image_path: str) -> tuple[str, str, list]:
    """
    Use Google Gemini Vision API for superior handwriting recognition.
    Returns: (text, method_used, word_confidences)
    """
    try:
        # Verify API key is configured
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            return None, None, []
        
        # Ensure API is configured
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            return None, None, []
        
        # Preprocess image first
        preprocessed_path = preprocess_image_for_ocr(image_path)
        
        # Load the image
        img = Image.open(preprocessed_path)
        
        # Get available Gemini model (uses cache)
        model_name, model = get_available_gemini_model()
        if not model:
            return None, None, []
        
        # Comprehensive prompt for best results
        prompt = """
        You are an expert at transcribing handwritten notes and documents. Please read and transcribe ALL handwritten text in this image carefully.
        
        CRITICAL INSTRUCTIONS:
        1. Read EVERY single word, letter, and number, even if unclear
        2. Preserve EXACT original structure: paragraphs, line breaks, indentation, lists
        3. If a word is ambiguous, use context and common sense - choose the most likely word
        4. Keep ALL numbers, dates, special characters, underscores, dashes, hyphens
        5. Preserve section headers and numbered lists (e.g., "Mod-1:", "1.", "2.", etc)
        6. If text is split across lines, join it properly with spaces
        7. Do NOT add any explanation, comments, or corrections
        8. Output ONLY the exact transcribed text - nothing else
        
        Handwritten text to transcribe:
        """
        
        # Generate content with timeout
        try:
            response = model.generate_content(
                [prompt, img],
                request_options={"timeout": 30}
            )
            
            # Check if response has text
            if not response or not hasattr(response, 'text'):
                raise ValueError("No response text received")
            
            text = response.text.strip()
            
        except Exception as api_error:
            # Only log actual errors, not quota/timeout issues
            error_str = str(api_error).lower()
            if "quota" not in error_str and "429" not in error_str and "timeout" not in error_str:
                print(f"[GEMINI] Error: {api_error}")
            return None, None, []
        
        # Clean up preprocessed image
        if preprocessed_path != image_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
            except:
                pass
        
        if text and len(text) > 5:
            confidences = estimate_confidences_from_text(text, default_conf=0.72)
            return text, "gemini_ai", confidences
        else:
            return None, None, []
            
    except Exception as e:
        print(f"[ERROR] Gemini OCR error: {e}")
        return None, None, []


def correct_spelling_with_ai(text: str) -> tuple[str, bool, str]:
    """
    Use Gemini AI to correct spelling and fix OCR errors.
    Returns: (corrected_text, success, error_message)
    """
    if not text or not text.strip():
        return text, False, "No text provided"
    
    try:
        # Verify API key is configured
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            return text, False, "API key not configured"
        
        # Ensure API is configured
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            return text, False, f"API configuration failed: {str(e)[:50]}"
        
        # Get available model
        model_name, model = get_available_gemini_model()
        if not model:
            return text, False, "No compatible Gemini model available"
        
        # Create prompt for spelling correction
        prompt = f"""You are an expert at correcting OCR (Optical Character Recognition) errors and fixing spelling mistakes in transcribed text.

TASK: Correct the following text that was extracted from an image using OCR. Fix:
1. Spelling errors and typos
2. Common OCR mistakes (e.g., '0' vs 'O', '1' vs 'I', 'rn' vs 'm')
3. Word boundaries that were incorrectly split or merged
4. Punctuation errors
5. Capitalization issues

IMPORTANT RULES:
- Preserve the original meaning and context
- Keep the same structure (paragraphs, line breaks, lists)
- Do NOT change proper nouns, names, or technical terms unless clearly wrong
- Do NOT add new content - only correct existing text
- Maintain numbers, dates, and special formatting
- If a word is ambiguous, choose the most likely correct word based on context

Original text with OCR errors:
{text}

Corrected text (output ONLY the corrected text, no explanations):"""
        
        # Generate corrected text
        try:
            response = model.generate_content(
                prompt,
                request_options={"timeout": 30}
            )
            
            if not response or not hasattr(response, 'text'):
                return text, False, "No response received from API"
            
            corrected_text = response.text.strip()
            
            if not corrected_text:
                return text, False, "Empty response from API"
            
            # Remove any markdown formatting if present
            if corrected_text.startswith("```"):
                lines = corrected_text.split("\n")
                corrected_text = "\n".join(lines[1:-1]) if len(lines) > 2 else corrected_text
            
            return corrected_text, True, ""
            
        except Exception as api_error:
            error_str = str(api_error).lower()
            if "quota" in error_str or "429" in error_str:
                return text, False, "API quota exceeded. Please try again later."
            elif "timeout" in error_str or "504" in error_str:
                return text, False, "Request timed out. Please try again."
            elif "400" in error_str:
                return text, False, f"Invalid request: {str(api_error)[:80]}"
            else:
                return text, False, f"API error: {str(api_error)[:80]}"
            
    except Exception as e:
        return text, False, f"Unexpected error: {str(e)[:80]}"


def run_easyocr_enhanced(image_path: str) -> tuple[str, str, list]:
    """Enhanced EasyOCR with preprocessing and confidences."""
    try:
        # Preprocess image
        preprocessed_path = preprocess_image_for_ocr(image_path)
        
        # Load preprocessed image
        img = cv2.imread(preprocessed_path)
        
        # Run EasyOCR
        results = reader.readtext(img, detail=1, paragraph=False)
        
        # Extract and sort text
        text_lines = []
        word_confidences = []
        for (bbox, text_val, confidence) in results:
            if confidence > 0.05:
                text_lines.append((bbox[0][1], text_val))
                for token in text_val.split():
                    word_confidences.append((token, float(confidence)))
        
        text_lines.sort(key=lambda x: x[0])
        text = "\n".join([t[1] for t in text_lines]).strip()

        # Clean up
        if preprocessed_path != image_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
            except:
                pass
        
        if text and len(text) > 5:
            return text, "easyocr_enhanced", word_confidences
        else:
            return None, None, []
            
    except Exception as e:
        return None, None, []


def run_tesseract_ocr(image_path: str) -> tuple[str, str, list]:
    """Tesseract OCR as fallback with confidences."""
    try:
        print("[OCR] Attempting Tesseract OCR...")
        
        preprocessed_path = preprocess_image_for_ocr(image_path)
        img = Image.open(preprocessed_path)
        
        text = pytesseract.image_to_string(
            img,
                lang="eng",
            config="--psm 3 --oem 1"
            ).strip()

        # Get word-level confidences
        word_confidences = []
        try:
            data = pytesseract.image_to_data(
                img,
                lang="eng",
                config="--psm 3 --oem 1",
                output_type=Output.DICT
            )
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                if not word or word.strip() == "":
                    continue
                try:
                    score = float(conf)
                    if score >= 0:
                        word_confidences.append((word, score / 100.0))
                except Exception:
                    continue
        except Exception as conf_error:
            print(f"[WARN] Tesseract confidence parse failed: {conf_error}")
        
        # Clean up
        if preprocessed_path != image_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
            except:
                pass
        
        if text and len(text) > 5:
            print(f"[SUCCESS] Tesseract succeeded! ({len(text)} characters)")
            return text, "tesseract", word_confidences
        else:
            return None, None, []
            
    except Exception as e:
        print(f"[ERROR] Tesseract error: {e}")
        return None, None, []


def refine_handwritten_text(text: str) -> str:
    """Advanced cleanup for handwritten OCR artifacts - aggressive post-processing."""
    if not text:
        return text
    
    import re
    
    # Step 1: Normalize unicode quotes/dashes
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2022": "•",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Step 2: Fix underscore separators (underscores between words should be spaces)
    # But preserve intentional underscores in code/variables
    text = re.sub(r"([a-zA-Z])_([a-zA-Z])", r"\1 \2", text)  # Replace underscores between letters with space
    
    # Step 3: Fix common separator patterns
    # Remove underscores used as dividers at line ends
    text = re.sub(r"_{2,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"~+$", "", text, flags=re.MULTILINE)
    
    # Step 4: Fix spacing issues
    text = re.sub(r"([a-zA-Z])([A-Z])", r"\1 \2", text)  # CamelCase to spaces
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)  # Letter followed by number
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)  # Number followed by letter
    
    # Step 5: Clean whitespace
    text = re.sub(r"[\t\r\f]+", " ", text)
    text = re.sub(r" {2,}", " ", text)  # Multiple spaces to single
    text = re.sub(r"\n\s+\n", "\n", text)  # Remove excessive blank lines
    
    # Step 6: Fix specific OCR confusions (prioritized by frequency in handwriting)
    confusions = [
        # Number-letter confusions
        (r"\bU1\b", "UI"),
        (r"\b1l\b", "ll"),
        (r"\bO0\b", "OO"),
        (r"\b0O\b", "OO"),
        (r"\blI\b", "ll"),
        (r"\bI1\b", "ll"),
        (r"\bVV\b", "W"),
        
        # Common words that get mangled
        (r"\bLLy\b", "Llly"),  # Keep proper nouns
        (r"\bJlu\b", "Jltu"),
        (r"\bdlsa\b", "dlsa"),
        (r"\bVlrtual", "Virtual"),
        (r"\bVlrtu", "Virtu"),
        (r"\bSerVlce", "Service"),
        (r"\bProVlde", "Provide"),
        (r"\bappllcation", "application"),
        (r"\bdesbtop", "desktop"),
        (r"\bnetuuark", "network"),
        (r"\bVirtuallsation", "Virtualisation"),
        (r"\bVlrtual", "Virtual"),
        (r"\bDiagraro", "Diagram"),
        (r"\bburid", "build"),
        (r"\bIdeptoy", "deploy"),
        (r"\bsccu", "scale"),
        (r"\bVlsualtsation", "Visualisation"),
    ]
    
    for pattern, replacement in confusions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Step 7: Fix list formatting
    # Ensure proper spacing after numbers/bullets
    text = re.sub(r"^(\d+)\.[\s]*", r"\1. ", text, flags=re.MULTILINE)
    text = re.sub(r"^([•*\-])[\s]*", r"\1 ", text, flags=re.MULTILINE)
    
    # Step 8: Clean up line breaks and structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        
        # Skip empty lines for now (will re-add selective blanks)
        if not line:
            continue
            
        # Capitalize first letter of sentences
        if line and not line[0].isupper() and line[0].isalpha():
            line = line[0].upper() + line[1:]
        
        cleaned_lines.append(line)
    
    # Rejoin with newlines
    text = '\n'.join(cleaned_lines)
    
    # Step 9: Add blank lines between major sections (detect headers like "Mod-1", "Mod-2", etc)
    text = re.sub(r'(Mod-\d+)', r'\n\n\1\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse excessive blank lines
    
    return text.strip()


def run_easyocr_basic(image_path: str) -> tuple[str, str, list]:
    """Basic EasyOCR WITHOUT preprocessing - standard mode."""
    try:
        print("[OCR] Running EasyOCR (Standard Mode - No AI)...")
        
        # Load image WITHOUT preprocessing
        img = cv2.imread(image_path)
        
        # Run EasyOCR with basic settings
        results = reader.readtext(img, detail=1, paragraph=False)
        
        # Extract text without sorting or filtering
        text_lines = []
        word_confidences = []
        for (bbox, text_val, confidence) in results:
            text_lines.append(text_val)
            for token in text_val.split():
                word_confidences.append((token, float(confidence)))
        
        text = "\n".join(text_lines).strip()
        
        if text and len(text) > 0:
            return text, "easyocr_standard", word_confidences
        else:
            return None, None, []

    except Exception as e:
        return None, None, []


def run_tesseract_basic(image_path: str) -> tuple[str, str, list]:
    """Basic Tesseract OCR WITHOUT preprocessing - standard mode."""
    try:
        # Load image WITHOUT preprocessing
        img = Image.open(image_path)
        
        # Run Tesseract with basic settings
        text = pytesseract.image_to_string(img, lang="eng").strip()
        word_confidences = []
        try:
            data = pytesseract.image_to_data(img, lang="eng", output_type=Output.DICT)
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                if word and conf and float(conf) >= 0:
                    word_confidences.append((word, float(conf) / 100.0))
        except Exception:
            pass
        
        if text and len(text) > 0:
            return text, "tesseract_standard", word_confidences
        else:
            return None, None, []
            
    except Exception as e:
        return None, None, []


def run_ocr(image_path: str, use_ai: bool = True) -> tuple[str, str, list]:
    """
    Multi-engine OCR with AI priority.
    Returns: (recognized_text, method_used, word_confidences)
    """
    methods_tried = []
    
    # AI MODE: Use Gemini + Enhanced OCR with preprocessing and text refinement
    if use_ai:
        # Try Gemini AI first (most accurate for handwriting)
        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE" and len(GEMINI_API_KEY.strip()) > 0:
            # Check API status first
            api_ok, api_msg = check_gemini_api_status()
            if api_ok:
                text, method, confidences = run_gemini_ocr(image_path)
                if text:
                    return refine_handwritten_text(text), method, confidences
                methods_tried.append("gemini_ai")
            else:
                methods_tried.append("gemini_ai")
        
        # Try EasyOCR Enhanced with preprocessing
        text, method, confidences = run_easyocr_enhanced(image_path)
        if text:
            return refine_handwritten_text(text), method, confidences
        methods_tried.append("easyocr_enhanced")
        
        # Try Tesseract with preprocessing
        text, method, confidences = run_tesseract_ocr(image_path)
        if text:
            return refine_handwritten_text(text), method, confidences
        methods_tried.append("tesseract_enhanced")
    
    # STANDARD MODE: Use basic OCR WITHOUT preprocessing or text refinement
    else:
        
        # Try EasyOCR basic (no preprocessing)
        text, method, confidences = run_easyocr_basic(image_path)
        if text:
            return text, method, confidences  # No refinement in standard mode
        methods_tried.append("easyocr_standard")
        
        # Try Tesseract basic (no preprocessing)
        text, method, confidences = run_tesseract_basic(image_path)
        if text:
            return text, method, confidences  # No refinement in standard mode
        methods_tried.append("tesseract_standard")
    
    return f"[OCR FAILED] Tried: {', '.join(methods_tried)}", "failed", []


def save_note(user_id: int, image_path: str, text: str, ocr_method: str = "unknown") -> int:
    """Save a note and return the note_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO notes (user_id, image_path, recognized_text, created_at, ocr_method)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, image_path, text, datetime.now().isoformat(timespec="seconds"), ocr_method))
    note_id = cur.lastrowid
    conn.commit()
    conn.close()
    return note_id

def get_notes_for_user(user_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT note_id, image_path, recognized_text, created_at, ocr_method
        FROM notes WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def update_note_text(note_id: int, new_text: str):
    """Update the recognized_text for an existing note."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE notes 
        SET recognized_text = ? 
        WHERE note_id = ?
    """, (new_text, note_id))
    conn.commit()
    conn.close()

def delete_note(note_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
    conn.commit()
    conn.close()

def export_to_pdf(text: str, filename: str) -> bool:
    """Export recognized text to PDF file."""
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor='#1a1a1a',
            spaceAfter=12,
            alignment=1
        )
        story.append(Paragraph("Handwritten Note", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666666',
            spaceAfter=12
        )
        story.append(Paragraph(f"Created: {timestamp}", date_style))
        story.append(Spacer(1, 0.2*inch))
        
        text_style = ParagraphStyle(
            'CustomText',
            parent=styles['BodyText'],
            fontSize=11,
            leading=14,
            spaceAfter=12
        )
        
        paragraphs = text.split('\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para, text_style))
            else:
                story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        return True
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        return False

# ---------------------- GUI LAYER ---------------------- #
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NoteVision AI - Handwritten Note Manager")
        self.geometry("1200x750")
        
        # Prevent window from closing unexpectedly
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Center window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Bring window to front
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(lambda: self.attributes('-topmost', False))
        
        self.current_user_id = None
        self.current_username = None
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        try:
            self.frames = {}
            for F in (LoginFrame, RegisterFrame, DashboardFrame):
                try:
                    frame = F(self, self)
                    self.frames[F] = frame
                    frame.grid(row=0, column=0, sticky="nsew")
                except Exception as frame_error:
                    print(f"[ERROR] Error creating frame {F.__name__}: {frame_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            self.show_frame(LoginFrame)
        except Exception as init_error:
            print(f"[ERROR] Error initializing frames: {init_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # Ensure window is visible and updated
        self.update()
        self.deiconify()
        print("[INFO] Window initialized and should be visible")
    
    def on_closing(self):
        """Handle window closing event."""
        print("[INFO] Window closing...")
        self.destroy()
    
    def show_frame(self, frame_class):
        frame = self.frames[frame_class]
        frame.tkraise()

class LoginFrame(ctk.CTkFrame):
    def __init__(self, parent, controller: App):
        super().__init__(parent, corner_radius=0)
        self.controller = controller
        
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        title_label = ctk.CTkLabel(
            center_frame, 
            text="🤖 NoteVision AI",
            font=ctk.CTkFont(size=42, weight="bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            center_frame,
            text="AI-Powered Handwriting Recognition",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 40))
        
        form_frame = ctk.CTkFrame(center_frame, width=400, corner_radius=15)
        form_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        login_title = ctk.CTkLabel(
            form_frame,
            text="Welcome Back",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        login_title.pack(pady=(30, 10))
        
        login_subtitle = ctk.CTkLabel(
            form_frame,
            text="Login to your account",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        login_subtitle.pack(pady=(0, 30))
        
        self.username_var = ctk.StringVar()
        username_label = ctk.CTkLabel(
            form_frame,
            text="Username",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        username_label.pack(fill="x", padx=40, pady=(0, 5))
        
        self.username_entry = ctk.CTkEntry(
            form_frame,
            textvariable=self.username_var,
            placeholder_text="Enter your username",
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.username_entry.pack(fill="x", padx=40, pady=(0, 20))
        
        self.password_var = ctk.StringVar()
        password_label = ctk.CTkLabel(
            form_frame,
            text="Password",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        password_label.pack(fill="x", padx=40, pady=(0, 5))
        
        self.password_entry = ctk.CTkEntry(
            form_frame,
            textvariable=self.password_var,
            placeholder_text="Enter your password",
            show="•",
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.password_entry.pack(fill="x", padx=40, pady=(0, 30))
        self.password_entry.bind("<Return>", lambda e: self.login_action())
        
        login_btn = ctk.CTkButton(
            form_frame,
            text="Login",
            command=self.login_action,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=10
        )
        login_btn.pack(fill="x", padx=40, pady=(0, 20))
        
        register_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        register_frame.pack(pady=(0, 30))
        
        ctk.CTkLabel(
            register_frame,
            text="Don't have an account?",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))
        
        register_btn = ctk.CTkButton(
            register_frame,
            text="Register",
            command=lambda: controller.show_frame(RegisterFrame),
            width=80,
            height=28,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            hover_color=("gray75", "gray25")
        )
        register_btn.pack(side="left")
    
    def login_action(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter username and password.")
            return
        
        success, msg, user_id = login_user(username, password)
        if not success:
            messagebox.showerror("Login Failed", msg)
            return
        
        self.controller.current_user_id = user_id
        self.controller.current_username = username
        self.controller.show_frame(DashboardFrame)
        self.controller.frames[DashboardFrame].refresh_notes()

class RegisterFrame(ctk.CTkFrame):
    def __init__(self, parent, controller: App):
        super().__init__(parent, corner_radius=0)
        self.controller = controller
        
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        title_label = ctk.CTkLabel(
            center_frame,
            text="🤖 NoteVision AI",
            font=ctk.CTkFont(size=42, weight="bold")
        )
        title_label.pack(pady=(0, 10))
        
        form_frame = ctk.CTkFrame(center_frame, width=400, corner_radius=15)
        form_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        register_title = ctk.CTkLabel(
            form_frame,
            text="Create Account",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        register_title.pack(pady=(30, 10))
        
        register_subtitle = ctk.CTkLabel(
            form_frame,
            text="Sign up to get started",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        register_subtitle.pack(pady=(0, 30))
        
        self.username_var = ctk.StringVar()
        username_label = ctk.CTkLabel(
            form_frame,
            text="Username",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        username_label.pack(fill="x", padx=40, pady=(0, 5))
        
        self.username_entry = ctk.CTkEntry(
            form_frame,
            textvariable=self.username_var,
            placeholder_text="Choose a username",
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.username_entry.pack(fill="x", padx=40, pady=(0, 15))
        
        self.email_var = ctk.StringVar()
        email_label = ctk.CTkLabel(
            form_frame,
            text="Email",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        email_label.pack(fill="x", padx=40, pady=(0, 5))
        
        self.email_entry = ctk.CTkEntry(
            form_frame,
            textvariable=self.email_var,
            placeholder_text="Enter your email",
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.email_entry.pack(fill="x", padx=40, pady=(0, 15))
        
        self.password_var = ctk.StringVar()
        password_label = ctk.CTkLabel(
            form_frame,
            text="Password",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        password_label.pack(fill="x", padx=40, pady=(0, 5))
        
        self.password_entry = ctk.CTkEntry(
            form_frame,
            textvariable=self.password_var,
            placeholder_text="Create a password (min 8 characters)",
            show="•",
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.password_entry.pack(fill="x", padx=40, pady=(0, 30))
        self.password_entry.bind("<Return>", lambda e: self.register_action())
        
        register_btn = ctk.CTkButton(
            form_frame,
            text="Create Account",
            command=self.register_action,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=10
        )
        register_btn.pack(fill="x", padx=40, pady=(0, 20))
        
        login_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        login_frame.pack(pady=(0, 30))
        
        ctk.CTkLabel(
            login_frame,
            text="Already have an account?",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))
        
        login_btn = ctk.CTkButton(
            login_frame,
            text="Login",
            command=lambda: controller.show_frame(LoginFrame),
            width=80,
            height=28,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            hover_color=("gray75", "gray25")
        )
        login_btn.pack(side="left")
    
    def register_action(self):
        username = self.username_var.get().strip()
        email = self.email_var.get().strip()
        password = self.password_var.get().strip()
        
        success, msg = register_user(username, email, password)
        if not success:
            messagebox.showerror("Registration Failed", msg)
            return
        
        messagebox.showinfo("Success", msg)
        self.controller.show_frame(LoginFrame)

class DashboardFrame(ctk.CTkFrame):
    def __init__(self, parent, controller: App):
        super().__init__(parent, corner_radius=0)
        self.controller = controller
        self.selected_image_path = None
        self.processing = False
        self.selected_note_id = None
        self.queue_data = {}
        self.task_queue = queue.Queue()
        self.processed_count = 0
        self.total_enqueued = 0
        self.stop_watch_event = threading.Event()
        self.watched_folder = None
        self.watching = False
        self.last_confidences = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        threading.Thread(target=self.queue_worker, daemon=True).start()
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Top bar
        top_frame = ctk.CTkFrame(self, height=70, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        top_frame.grid_columnconfigure(1, weight=1)
        
        title_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w", padx=20, pady=15)
        
        ctk.CTkLabel(
            title_frame,
            text="🤖 NoteVision AI",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(side="left", padx=(0, 20))
        
        self.welcome_label = ctk.CTkLabel(
            title_frame,
            text="Welcome",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.welcome_label.pack(side="left")
        
        btn_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e", padx=20, pady=15)
        
        # AI Toggle Switch
        self.use_ai_var = ctk.BooleanVar(value=True)
        self.ai_switch = ctk.CTkSwitch(
            btn_frame,
            text="🤖 AI Mode",
            variable=self.use_ai_var,
            font=ctk.CTkFont(size=13),
            onvalue=True,
            offvalue=False,
            command=self.update_ai_indicator
        )
        self.ai_switch.pack(side="left", padx=5)
        
        # AI Status Button (NEW!)
        self.status_btn = ctk.CTkButton(
            btn_frame,
            text="🔍 Check AI",
            command=self.show_ai_status,
            width=100,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="#2196F3",
            hover_color="#1976D2",
            corner_radius=8
        )
        self.status_btn.pack(side="left", padx=5)
        
        self.theme_btn = ctk.CTkButton(
            btn_frame,
            text="🌙 Dark",
            command=self.toggle_theme,
            width=100,
            height=35,
            font=ctk.CTkFont(size=13),
            corner_radius=8
        )
        self.theme_btn.pack(side="left", padx=5)
        
        logout_btn = ctk.CTkButton(
            btn_frame,
            text="Logout",
            command=self.logout,
            width=100,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            corner_radius=8
        )
        logout_btn.pack(side="left", padx=5)
        
        # Main content
        content = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        
        # Left panel (scrollable so the OCR button and results remain reachable on small screens)
        left_panel = ctk.CTkScrollableFrame(content, corner_radius=15)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.grid_rowconfigure(6, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            left_panel,
            text="Upload & Process",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=(20, 10), padx=20, sticky="w")
        
        upload_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        upload_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        upload_frame.grid_columnconfigure(0, weight=1)
        
        self.select_btn = ctk.CTkButton(
            upload_frame,
            text="📁 Select Image",
            command=self.select_image,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10
        )
        self.select_btn.grid(row=0, column=0, sticky="ew", pady=5)

        folder_btn = ctk.CTkButton(
            upload_frame,
            text="📂 Process Folder",
            command=self.process_folder,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10,
            fg_color="#607D8B",
            hover_color="#546E7A"
        )
        folder_btn.grid(row=1, column=0, sticky="ew", pady=5)

        self.watch_btn = ctk.CTkButton(
            upload_frame,
            text="👀 Watch Folder",
            command=self.toggle_folder_watch,
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            fg_color="#37474F",
            hover_color="#263238"
        )
        self.watch_btn.grid(row=2, column=0, sticky="ew", pady=5)
        
        self.image_label = ctk.CTkLabel(
            upload_frame,
            text="No image selected",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.image_label.grid(row=3, column=0, pady=5, sticky="w")

        self.auto_save_var = ctk.BooleanVar(value=True)
        auto_save_check = ctk.CTkCheckBox(
            upload_frame,
            text="Auto-save results to DB",
            variable=self.auto_save_var,
            font=ctk.CTkFont(size=12)
        )
        auto_save_check.grid(row=4, column=0, sticky="w", pady=(0, 5))

        self.batch_progress = ctk.CTkProgressBar(upload_frame, height=12)
        self.batch_progress.set(0)
        self.batch_progress.grid(row=5, column=0, sticky="ew", pady=(2, 6))

        self.queue_frame = ctk.CTkScrollableFrame(upload_frame, height=140, corner_radius=8)
        self.queue_frame.grid(row=6, column=0, sticky="nsew", pady=(4, 6))
        self.queue_frame.grid_columnconfigure(0, weight=1)
        self.queue_widgets = {}
        
        self.preview_frame = ctk.CTkFrame(left_panel, height=220, corner_radius=10)
        self.preview_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.preview_frame.grid_propagate(False)
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="📷\n\nImage Preview",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.preview_label.pack(expand=True, fill="both")
        self.preview_photo = None
        
        # OCR Method Label
        self.ocr_method_label = ctk.CTkLabel(
            left_panel,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.ocr_method_label.grid(row=3, column=0, padx=20, pady=(0, 5), sticky="w")
        
        self.ocr_btn = ctk.CTkButton(
            left_panel,
            text="🤖 AI Scan & Save",
            command=self.ocr_and_save,
            height=50,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#4CAF50",
            hover_color="#45a049",
            corner_radius=10
        )
        self.ocr_btn.grid(row=4, column=0, sticky="ew", padx=20, pady=10)
        
        text_header = ctk.CTkFrame(left_panel, fg_color="transparent")
        text_header.grid(row=5, column=0, sticky="ew", padx=20, pady=(10, 5))
        text_header.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            text_header,
            text="Recognized Text",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.low_conf_label = ctk.CTkLabel(
            text_header,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#FFB300"
        )
        self.low_conf_label.grid(row=1, column=0, sticky="w")
        
        text_container = ctk.CTkFrame(left_panel, fg_color="transparent")
        text_container.grid(row=6, column=0, sticky="nsew", padx=20, pady=(0, 10))
        text_container.grid_rowconfigure(0, weight=1)
        text_container.grid_columnconfigure(0, weight=1)
        
        self.text_box = ctk.CTkTextbox(
            text_container,
            font=ctk.CTkFont(size=13),
            corner_radius=10,
            wrap="word"
        )
        self.text_box.grid(row=0, column=0, sticky="nsew")
        
        action_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        action_frame.grid(row=7, column=0, sticky="ew", padx=20, pady=(0, 20))
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)
        action_frame.grid_columnconfigure(2, weight=1)
        action_frame.grid_columnconfigure(3, weight=1)
        
        copy_btn = ctk.CTkButton(
            action_frame,
            text="📋 Copy",
            command=self.copy_text,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#FF9800",
            hover_color="#F57C00",
            corner_radius=8
        )
        copy_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        spell_btn = ctk.CTkButton(
            action_frame,
            text="✨ Fix Spelling",
            command=self.correct_spelling,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#4CAF50",
            hover_color="#45a049",
            corner_radius=8
        )
        spell_btn.grid(row=0, column=1, sticky="ew", padx=5)
        
        pdf_btn = ctk.CTkButton(
            action_frame,
            text="📄 Export PDF",
            command=self.download_pdf,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#9C27B0",
            hover_color="#7B1FA2",
            corner_radius=8
        )
        pdf_btn.grid(row=0, column=2, sticky="ew", padx=5)
        
        save_btn = ctk.CTkButton(
            action_frame,
            text="💾 Save",
            command=self.save_current_text,
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#2196F3",
            hover_color="#1976D2",
            corner_radius=8
        )
        save_btn.grid(row=0, column=3, sticky="ew", padx=(5, 0))

        summarize_btn = ctk.CTkButton(
            action_frame,
            text="📝 Summarize",
            command=self.summarize_text,
            height=38,
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            fg_color="#26A69A",
            hover_color="#1e8f85"
        )
        summarize_btn.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=6)

        keywords_btn = ctk.CTkButton(
            action_frame,
            text="🔑 Keywords",
            command=self.extract_keywords,
            height=38,
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            fg_color="#5C6BC0",
            hover_color="#3F51B5"
        )
        keywords_btn.grid(row=1, column=1, sticky="ew", padx=5, pady=6)
        
        # Right panel
        right_panel = ctk.CTkFrame(content, corner_radius=15)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_panel.grid_rowconfigure(2, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        header_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            header_frame,
            text="Your Notes",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w")
        
        self.notes_count_label = ctk.CTkLabel(
            header_frame,
            text="0 notes",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.notes_count_label.grid(row=0, column=1, sticky="e")
        
        search_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        search_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        
        self.search_var = ctk.StringVar()
        self.search_entry = ctk.CTkEntry(
            search_frame,
            textvariable=self.search_var,
            placeholder_text="🔍 Search notes...",
            height=40,
            font=ctk.CTkFont(size=13),
            corner_radius=10
        )
        self.search_entry.pack(fill="x")
        self.search_entry.bind("<KeyRelease>", self.filter_notes)
        
        notes_container = ctk.CTkScrollableFrame(
            right_panel,
            corner_radius=10,
            fg_color="transparent"
        )
        notes_container.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 10))
        notes_container.grid_columnconfigure(0, weight=1)
        
        self.notes_list_frame = notes_container
        self.note_widgets = []
        
        self.delete_btn = ctk.CTkButton(
            right_panel,
            text="🗑️ Delete Selected",
            command=self.delete_selected,
            height=45,
            font=ctk.CTkFont(size=14),
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            corner_radius=10,
            state="disabled"
        )
        self.delete_btn.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        # Initial AI status check
        self.after(1000, self.update_ai_indicator)
    
    def show_ai_status(self):
        """Show detailed AI engine status - THIS IS THE NEW METHOD!"""
        status = get_ocr_engine_status()
        
        # Create status window
        status_window = ctk.CTkToplevel(self)
        status_window.title("OCR Engine Status")
        status_window.geometry("500x400")
        status_window.transient(self)
        status_window.grab_set()
        
        # Center window
        status_window.update_idletasks()
        x = (status_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (status_window.winfo_screenheight() // 2) - (400 // 2)
        status_window.geometry(f'500x400+{x}+{y}')
        
        # Title
        ctk.CTkLabel(
            status_window,
            text="🤖 OCR Engine Status",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=20)
        
        # Status cards
        status_frame = ctk.CTkFrame(status_window, fg_color="transparent")
        status_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        engines = [
            ("Gemini AI", "gemini", "🤖", "Best for handwriting"),
            ("EasyOCR", "easyocr", "🔍", "Good for printed text"),
            ("Tesseract", "tesseract", "📝", "Basic OCR fallback")
        ]
        
        for idx, (name, key, icon, desc) in enumerate(engines):
            engine_status = status[key]
            available = engine_status['available']
            message = engine_status['message']
            
            card = ctk.CTkFrame(
                status_frame,
                corner_radius=10,
                fg_color=("#e8f5e9" if available else "#ffebee") if ctk.get_appearance_mode() == "Light" else ("#1b5e20" if available else "#b71c1c")
            )
            card.pack(fill="x", pady=8)
            
            # Header
            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=20, pady=(15, 5))
            
            ctk.CTkLabel(
                header,
                text=f"{icon} {name}",
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="w"
            ).pack(side="left")
            
            status_badge = ctk.CTkLabel(
                header,
                text="✓ Active" if available else "✗ Unavailable",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white",
                fg_color="#4CAF50" if available else "#f44336",
                corner_radius=5,
                padx=10,
                pady=3
            )
            status_badge.pack(side="right")
            
            # Description
            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=12),
                text_color="gray",
                anchor="w"
            ).pack(fill="x", padx=20, pady=(0, 5))
            
            # Status message
            ctk.CTkLabel(
                card,
                text=f"Status: {message}",
                font=ctk.CTkFont(size=11),
                anchor="w"
            ).pack(fill="x", padx=20, pady=(0, 15))
        
        # Current mode indicator
        mode_frame = ctk.CTkFrame(status_window, corner_radius=10)
        mode_frame.pack(fill="x", padx=30, pady=10)
        
        current_mode = "AI Mode" if self.use_ai_var.get() else "Standard Mode"
        mode_color = "#4CAF50" if self.use_ai_var.get() else "#FF9800"
        
        ctk.CTkLabel(
            mode_frame,
            text=f"Current Mode: {current_mode}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=mode_color
        ).pack(pady=15)
        
        # Close button
        ctk.CTkButton(
            status_window,
            text="Close",
            command=status_window.destroy,
            width=120,
            height=35,
            font=ctk.CTkFont(size=13),
            corner_radius=8
        ).pack(pady=15)
    
    def update_ai_indicator(self):
        """Update AI mode indicator in real-time."""
        if self.use_ai_var.get():
            status = get_ocr_engine_status()
            if status['gemini']['available']:
                self.ocr_btn.configure(
                    text="🤖 AI Scan & Save",
                    fg_color="#4CAF50",
                    hover_color="#45a049"
                )
            else:
                self.ocr_btn.configure(
                    text="🔍 Enhanced Scan & Save",
                    fg_color="#2196F3",
                    hover_color="#1976D2"
                )
        else:
            self.ocr_btn.configure(
                text="🔍 Standard Scan & Save",
                fg_color="#2196F3",
                hover_color="#1976D2"
            )

    def queue_worker(self):
        """Background worker that dispatches queued files into OCR threads."""
        while True:
            path = self.task_queue.get()
            if path is None:
                break
            future = self.executor.submit(self.process_single_file, path)
            future.add_done_callback(lambda _f: self.task_queue.task_done())

    def _create_queue_widget(self, path: str):
        row = ctk.CTkFrame(self.queue_frame, corner_radius=6)
        row.grid_columnconfigure(0, weight=1)
        row.pack(fill="x", pady=2)
        name_label = ctk.CTkLabel(row, text=os.path.basename(path), anchor="w")
        name_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        status_label = ctk.CTkLabel(row, text="Queued", text_color="gray")
        status_label.grid(row=0, column=1, sticky="e", padx=6, pady=4)
        self.queue_widgets[path] = (row, status_label)

    def _update_queue_status(self, path: str, status: str, message: str = ""):
        widget = self.queue_widgets.get(path)
        if not widget:
            return
        _, status_label = widget
        colors = {
            "queued": ("Queued", "gray"),
            "processing": ("Processing", "#03A9F4"),
            "done": ("Done", "#4CAF50"),
            "failed": ("Failed", "#f44336"),
        }
        label, color = colors.get(status, (status, "gray"))
        status_label.configure(text=f"{label} {message}", text_color=color)

    def _update_progress(self):
        total = self.total_enqueued if self.total_enqueued else 1
        progress = min(1.0, self.processed_count / total)
        self.batch_progress.set(progress)

    def enqueue_files(self, paths: list[str]):
        new_paths = [p for p in paths if p not in self.queue_data]
        if not new_paths:
            return
        for p in new_paths:
            self.queue_data[p] = {"status": "queued"}
            self.after(0, lambda p=p: self._create_queue_widget(p))
            self.task_queue.put(p)
            self.total_enqueued += 1
        self._update_progress()

    def process_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ]
        if not images:
            messagebox.showinfo("No images", "No supported images found in this folder.")
            return
        self.enqueue_files(images)
        messagebox.showinfo("Queued", f"Queued {len(images)} file(s) for OCR.")

    def toggle_folder_watch(self):
        if not self.watching:
            folder = filedialog.askdirectory(title="Watch folder for new images")
            if not folder:
                return
            self.watched_folder = folder
            self.stop_watch_event.clear()
            threading.Thread(target=self.folder_watch_loop, daemon=True).start()
            self.watching = True
            self.watch_btn.configure(text="⏹ Stop Watching")
        else:
            self.stop_watch_event.set()
            self.watching = False
            self.watch_btn.configure(text="👀 Watch Folder")

    def folder_watch_loop(self):
        seen = set()
        while not self.stop_watch_event.is_set() and self.watched_folder:
            try:
                for name in os.listdir(self.watched_folder):
                    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                        continue
                    full = os.path.join(self.watched_folder, name)
                    if full not in seen:
                        seen.add(full)
                        self.enqueue_files([full])
                time.sleep(3)
            except Exception as e:
                print(f"[WATCH] Error watching folder: {e}")
                time.sleep(5)

    def process_single_file(self, path: str):
        self.after(0, lambda: self._update_queue_status(path, "processing"))
        try:
            text, method, confidences = run_ocr(path, use_ai=self.use_ai_var.get())
            if not text or "[OCR FAILED]" in text:
                self.after(0, lambda: self._update_queue_status(path, "failed", "(no text)"))
                return
            if self.auto_save_var.get() and self.controller.current_user_id:
                note_id = save_note(self.controller.current_user_id, path, text, method)
                # Set selected note if this is the currently displayed file
                if path == self.selected_image_path:
                    self.selected_note_id = note_id
            self.after(0, lambda: self._update_queue_status(path, "done", method))
        except Exception as e:
            self.after(0, lambda: self._update_queue_status(path, "failed", str(e)))
        finally:
            self.processed_count += 1
            self.after(0, self._update_progress)
    
    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            ctk.set_appearance_mode("Light")
            self.theme_btn.configure(text="☀️ Light")
        else:
            ctk.set_appearance_mode("Dark")
            self.theme_btn.configure(text="🌙 Dark")
    
    def refresh_notes(self):
        if self.controller.current_username:
            self.welcome_label.configure(text=f"Welcome, {self.controller.current_username}!")
        
        for widget in self.note_widgets:
            widget.destroy()
        self.note_widgets.clear()
        
        if not self.controller.current_user_id:
            return
        
        notes = get_notes_for_user(self.controller.current_user_id)
        self.notes_count_label.configure(text=f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
        
        for note in notes:
            note_id, image_path, recognized_text, created_at, ocr_method = note
            self.create_note_card(note_id, recognized_text, created_at, ocr_method)
    
    def create_note_card(self, note_id, text, date, ocr_method="unknown"):
        card = ctk.CTkFrame(
            self.notes_list_frame,
            corner_radius=10,
            fg_color=("gray85", "gray20"),
            cursor="hand2"
        )
        card.grid(sticky="ew", pady=5, padx=2)
        card.grid_columnconfigure(0, weight=1)
        card.note_id = note_id
        
        card.bind("<Button-1>", lambda e, nid=note_id: self.select_note_card(nid))
        
        # Header with date and method
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 5))
        header_frame.grid_columnconfigure(0, weight=1)
        
        date_label = ctk.CTkLabel(
            header_frame,
            text=date,
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        )
        date_label.grid(row=0, column=0, sticky="w")
        date_label.bind("<Button-1>", lambda e, nid=note_id: self.select_note_card(nid))
        
        # Method badge
        method_colors = {
            "gemini_ai": "#4CAF50",
            "easyocr_enhanced": "#2196F3",
            "tesseract": "#FF9800",
            "unknown": "gray"
        }
        method_names = {
            "gemini_ai": "🤖 AI",
            "easyocr_enhanced": "🔍 Enhanced",
            "tesseract": "📝 Basic",
            "unknown": "?"
        }
        
        method_label = ctk.CTkLabel(
            header_frame,
            text=method_names.get(ocr_method, "?"),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="white",
            fg_color=method_colors.get(ocr_method, "gray"),
            corner_radius=5,
            padx=8,
            pady=2
        )
        method_label.grid(row=0, column=1, sticky="e")
        method_label.bind("<Button-1>", lambda e, nid=note_id: self.select_note_card(nid))
        
        preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
        preview_label = ctk.CTkLabel(
            card,
            text=preview,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left"
        )
        preview_label.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 12))
        preview_label.bind("<Button-1>", lambda e, nid=note_id: self.select_note_card(nid))
        
        self.note_widgets.append(card)
    
    def select_note_card(self, note_id):
        for widget in self.note_widgets:
            if hasattr(widget, 'note_id'):
                if widget.note_id == note_id:
                    widget.configure(fg_color=("gray75", "gray30"), border_width=2, border_color=("#1f6aa5", "#1f6aa5"))
                    self.selected_note_id = note_id
                else:
                    widget.configure(fg_color=("gray85", "gray20"), border_width=0)
        
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT recognized_text, image_path, ocr_method FROM notes WHERE note_id = ?", (note_id,))
        row = cur.fetchone()
        conn.close()
        
        if row:
            text = row[0]
            image_path = row[1]
            ocr_method = row[2] if len(row) > 2 else "unknown"
            
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", text)
            
            self.selected_image_path = image_path
            self.image_label.configure(text=f"Selected: {os.path.basename(image_path)}")
            
            # Show OCR method
            method_names = {
                "gemini_ai": "Scanned with: Gemini AI 🤖",
                "easyocr_enhanced": "Scanned with: EasyOCR Enhanced 🔍",
                "tesseract": "Scanned with: Tesseract 📝",
                "unknown": ""
            }
            self.ocr_method_label.configure(text=method_names.get(ocr_method, ""))
            
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    preview_size = (400, 200)
                    img.thumbnail(preview_size, Image.Resampling.LANCZOS)
                    
                    self.preview_photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                    self.preview_label.configure(image=self.preview_photo, text="")
                else:
                    self.preview_label.configure(image=None, text="📷\n\nImage not found")
                    self.preview_photo = None
            except Exception as e:
                self.preview_label.configure(image=None, text=f"📷\n\nPreview Error:\n{str(e)}")
                self.preview_photo = None
            
            self.delete_btn.configure(state="normal")

    def apply_confidence_highlighting(self, confidences: list, threshold: float = 0.65):
        """Apply confidence highlighting - CTkTextbox doesn't support tags, so we just count low-confidence words."""
        try:
            # CTkTextbox doesn't support tag_configure/tag_add, so we can't highlight
            # Instead, just count and display low-confidence words
            text = self.text_box.get("1.0", "end-1c")
            low_count = 0
            for word, conf in confidences:
                if conf < threshold and word in text:
                    low_count += 1
            
            label_text = f"Low-confidence words: {low_count}" if low_count else ""
            if hasattr(self, 'low_conf_label'):
                self.low_conf_label.configure(text=label_text)
        except Exception as e:
            # Silently fail if highlighting isn't supported
            pass
    
    def filter_notes(self, event=None):
        search_term = self.search_var.get().lower()
        
        for widget in self.note_widgets:
            widget.destroy()
        self.note_widgets.clear()
        
        if not self.controller.current_user_id:
            return
        
        notes = get_notes_for_user(self.controller.current_user_id)
        
        filtered_notes = [
            note for note in notes
            if search_term in note[2].lower() or search_term in note[3].lower()
        ]
        
        self.notes_count_label.configure(
            text=f"{len(filtered_notes)} note{'s' if len(filtered_notes) != 1 else ''}"
        )
        
        for note in filtered_notes:
            note_id, image_path, recognized_text, created_at, ocr_method = note
            self.create_note_card(note_id, recognized_text, created_at, ocr_method)
    
    def select_image(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if path:
            self.selected_image_path = path
            self.image_label.configure(text=f"Selected: {os.path.basename(path)}")
            self.ocr_method_label.configure(text="")
            
            try:
                img = Image.open(path)
                preview_size = (400, 200)
                img.thumbnail(preview_size, Image.Resampling.LANCZOS)
                
                self.preview_photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self.preview_label.configure(image=self.preview_photo, text="")
            except Exception as e:
                self.preview_label.configure(image=None, text=f"📷\n\nPreview Error:\n{str(e)}")
                self.preview_photo = None
    
    def ocr_and_save(self):
        if not self.selected_image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return
        
        if self.processing:
            messagebox.showwarning("Processing", "OCR is already running.")
            return
        
        use_ai = self.use_ai_var.get()
        
        def process():
            self.processing = True
            btn_text = "🤖 AI Processing..." if use_ai else "⏳ Processing..."
            self.after(0, lambda: self.ocr_btn.configure(text=btn_text, state="disabled"))
            
            try:
                text, method, confidences = run_ocr(self.selected_image_path, use_ai=use_ai)
                
                if not text.strip() or "[OCR FAILED]" in text:
                    self.after(0, lambda: messagebox.showwarning("No Text", f"OCR did not detect any text.\n{text}"))
                    return
                
                self.text_box.delete("1.0", "end")
                self.text_box.insert("1.0", text)
                self.last_confidences = confidences or []
                self.apply_confidence_highlighting(self.last_confidences)
                
                # Show method used
                method_names = {
                    "gemini_ai": "Scanned with: Gemini AI 🤖 (Highest Accuracy)",
                    "easyocr_enhanced": "Scanned with: EasyOCR Enhanced 🔍 (AI Mode)",
                    "tesseract_enhanced": "Scanned with: Tesseract Enhanced 📝 (AI Mode)",
                    "easyocr_standard": "Scanned with: EasyOCR 🔍 (Standard Mode)",
                    "tesseract_standard": "Scanned with: Tesseract 📝 (Standard Mode)",
                    "tesseract": "Scanned with: Tesseract 📝",
                    "failed": "OCR Failed"
                }
                method_val = method
                method_display = method_names.get(method_val, f"Method: {method_val}")
                self.after(0, lambda m=method_display: self.ocr_method_label.configure(text=m))
                
                if self.auto_save_var.get():
                    note_id = save_note(self.controller.current_user_id, self.selected_image_path, text, method)
                    # Set the selected note so user can save changes later
                    self.selected_note_id = note_id
                success_msg = f"Note processed!\n\nMethod used: {method_names.get(method_val, method_val)}"
                self.after(0, lambda msg=success_msg: messagebox.showinfo("Success", msg))
                self.after(0, self.refresh_notes)
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"OCR failed: {msg}"))
            finally:
                self.processing = False
                btn_text = "🤖 AI Scan & Save" if use_ai else "🔍 Standard Scan & Save"
                self.after(0, lambda: self.ocr_btn.configure(text=btn_text, state="normal"))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def delete_selected(self):
        if not self.selected_note_id:
            messagebox.showwarning("Select a note", "Please select a note to delete.")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this note?"):
            delete_note(self.selected_note_id)
            self.text_box.delete("1.0", "end")
            self.selected_note_id = None
            self.delete_btn.configure(state="disabled")
            self.ocr_method_label.configure(text="")
            self.refresh_notes()
            messagebox.showinfo("Deleted", "Note deleted successfully.")
    
    def copy_text(self):
        text = self.text_box.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("Empty", "No text to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Text copied to clipboard!")
    
    def save_current_text(self):
        """Save the current text in the text box to the selected note."""
        if not hasattr(self, 'selected_note_id') or not self.selected_note_id:
            messagebox.showwarning(
                "No Note Selected",
                "Please select a note from 'Your Notes' first to save changes."
            )
            return
        
        text = self.text_box.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("Empty", "No text to save.")
            return
        
        try:
            update_note_text(self.selected_note_id, text)
            messagebox.showinfo("Saved", "Text saved successfully!")
            # Refresh notes list to show updated preview
            self.refresh_notes()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def correct_spelling(self):
        """Use AI to correct spelling and fix OCR errors."""
        text = self.text_box.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("Empty", "No text to correct. Run OCR first.")
            return
        
        # Check if API is available
        api_ok, api_msg = check_gemini_api_status()
        if not api_ok:
            messagebox.showerror(
                "AI Unavailable",
                f"Cannot correct spelling: {api_msg}\n\nPlease check your API configuration."
            )
            return
        
        # Show processing message
        original_text = text
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", "Correcting spelling with AI... Please wait...")
        self.update()
        
        def process_correction():
            try:
                corrected_text, success, error_msg = correct_spelling_with_ai(original_text)
                
                if success:
                    # Save to database if a note is selected
                    if hasattr(self, 'selected_note_id') and self.selected_note_id:
                        try:
                            update_note_text(self.selected_note_id, corrected_text)
                            save_msg = "Spelling corrected and saved to database!"
                        except Exception as save_error:
                            save_msg = f"Spelling corrected, but failed to save: {str(save_error)[:50]}"
                    else:
                        save_msg = "Spelling corrected successfully!\n\nNote: Text not saved (no note selected)."
                    
                    self.after(0, lambda: self.text_box.delete("1.0", "end"))
                    self.after(0, lambda: self.text_box.insert("1.0", corrected_text))
                    self.after(0, lambda msg=save_msg: messagebox.showinfo("Success", msg))
                    
                    # Refresh notes list to show updated text
                    if hasattr(self, 'selected_note_id') and self.selected_note_id:
                        self.after(0, self.refresh_notes)
                else:
                    self.after(0, lambda: self.text_box.delete("1.0", "end"))
                    self.after(0, lambda: self.text_box.insert("1.0", original_text))
                    error_display = error_msg if error_msg else "Failed to correct spelling. Please try again or check your API connection."
                    self.after(0, lambda msg=error_display: messagebox.showerror("Error", msg))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self.text_box.delete("1.0", "end"))
                self.after(0, lambda: self.text_box.insert("1.0", original_text))
                self.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Spelling correction failed: {msg}"))
        
        thread = threading.Thread(target=process_correction, daemon=True)
        thread.start()

    def summarize_text(self):
        text = self.text_box.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("Empty", "No text to summarize.")
            return
        api_ok, api_msg = check_gemini_api_status()
        if not api_ok:
            messagebox.showerror("AI Unavailable", api_msg)
            return

        def run_summary():
            try:
                model_name, model = get_available_gemini_model()
                if not model:
                    raise RuntimeError("No Gemini model available")
                prompt = f"Summarize these handwritten notes into 4-6 concise bullet points with key facts and dates preserved:\n\n{text}\n\nSummary:"  # noqa: E501
                resp = model.generate_content(prompt, request_options={"timeout": 25})
                summary = resp.text.strip() if hasattr(resp, "text") else ""
                if summary:
                    def update_ui():
                        self.text_box.insert("end", "\n\n--- Summary ---\n" + summary + "\n")
                        messagebox.showinfo("Summary", "Summary added to the text box.")
                    self.after(0, update_ui)
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Summarization failed: {msg}"))

        threading.Thread(target=run_summary, daemon=True).start()

    def extract_keywords(self):
        text = self.text_box.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showwarning("Empty", "No text to process.")
            return
        api_ok, api_msg = check_gemini_api_status()
        if not api_ok:
            messagebox.showerror("AI Unavailable", api_msg)
            return

        def run_keywords():
            try:
                model_name, model = get_available_gemini_model()
                if not model:
                    raise RuntimeError("No Gemini model available")
                prompt = (
                    "Extract up to 12 key terms, people, dates, and action items from these notes. "
                    "Return a comma-separated list only.\n\n" + text
                )
                resp = model.generate_content(prompt, request_options={"timeout": 20})
                keywords = resp.text.strip() if hasattr(resp, "text") else ""
                if keywords:
                    def update_ui():
                        self.text_box.insert("end", "\n\n--- Keywords ---\n" + keywords + "\n")
                        messagebox.showinfo("Keywords", "Keywords appended to the text box.")
                    self.after(0, update_ui)
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Keyword extraction failed: {msg}"))

        threading.Thread(target=run_keywords, daemon=True).start()
    
    def download_pdf(self):
        text = self.text_box.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showerror("Error", "No text to download. Run OCR first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if not file_path:
            return
        
        if export_to_pdf(text, file_path):
            messagebox.showinfo("Success", f"PDF saved successfully!")
        else:
            messagebox.showerror("Error", "Failed to create PDF file.")
    
    def logout(self):
        self.controller.current_user_id = None
        self.controller.current_username = None
        self.selected_image_path = None
        self.selected_note_id = None
        self.image_label.configure(text="No image selected")
        self.text_box.delete("1.0", "end")
        self.preview_label.configure(image=None, text="📷\n\nImage Preview")
        self.preview_photo = None
        self.delete_btn.configure(state="disabled")
        self.ocr_method_label.configure(text="")
        
        for widget in self.note_widgets:
            widget.destroy()
        self.note_widgets.clear()
        
        self.controller.show_frame(LoginFrame)

# ---------------------- MAIN ENTRY ---------------------- #
if __name__ == "__main__":
    import sys
    try:
        print("[INFO] Initializing database...")
        init_db()
        print("[INFO] Database initialized")
        print("[INFO] Creating application window...")
        app = App()
        print("[INFO] Application window created")
        print("[INFO] Window should be visible now...")
        print("[INFO] Starting main loop (window should stay open)...")
        print("[INFO] If window doesn't appear, check Task Manager for python.exe process")
        try:
            # Force window to front one more time
            app.after(100, lambda: app.lift())
            app.after(100, lambda: app.focus_force())
            app.mainloop()
        except KeyboardInterrupt:
            print("[INFO] Application interrupted by user")
        except Exception as loop_error:
            print(f"[ERROR] Error in main loop: {loop_error}")
            import traceback
            traceback.print_exc()
        print("[INFO] Application closed")
    except Exception as e:
        import traceback
        print(f"[ERROR] Error starting application: {e}")
        traceback.print_exc()
        print("\n[INFO] Press Enter to exit...")
        try:
            input()
        except:
            pass
        sys.exit(1)
