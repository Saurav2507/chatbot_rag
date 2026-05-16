import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re

def clean_text(text: str) -> str:
    # Remove excessive newlines and normalize spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def process_pdf(file_path: str) -> list[dict]:
    """
    Extracts text from PDF page by page.
    Uses native text extraction first. If text length < 50 chars, uses OCR.
    """
    doc = fitz.open(file_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Try native text extraction
        text = page.get_text("text")
        extraction_mode = "native"

        # Fallback to OCR if text is very short or missing
        if len(text.strip()) < 50:
            extraction_mode = "ocr"
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            
        text = clean_text(text)
        
        if len(text) > 10: # Only keep pages with some actual content
            pages_data.append({
                "page_number": page_num + 1, # 1-indexed
                "text": text,
                "extraction_mode": extraction_mode
            })
            
    return pages_data
