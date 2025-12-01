import fitz
import os
import json
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed

def extract_images(pdf_path, output_img_folder, output_json):
    os.makedirs(output_img_folder, exist_ok=True)
    results = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for idx, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            img_filename = f"page_{page_num+1}_img_{idx+1}.png"
            img_save_path = os.path.join(output_img_folder, img_filename)

            if pix.n < 5:
                pix.save(img_save_path)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_save_path)

            # OCR
            ocr_text = pytesseract.image_to_string(Image.open(img_save_path))

            results.append({
                "page": page_num + 1,
                "image_file": img_filename,
                "ocr": ocr_text.strip(),
                "type": "image"
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Image + OCR extraction complete.")
    print(f"Total images extracted: {len(results)}")


if __name__ == "__main__":
    extract_images(
        pdf_path="data/raw/qatar_test_doc.pdf",
        output_img_folder="data/images",
        output_json="data/processed/image_ocr.json"
    )
