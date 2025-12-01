import fitz  # PyMuPDF
import json
import os

def extract_text_from_pdf(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        chunk = {
            "page": page_num + 1,
            "content": text.strip(),
            "type": "text"
        }

        results.append(chunk)

    # Save to jsonl format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Text extraction completed. Total pages: {len(doc)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    pdf_path = "data/raw/qatar_test_doc.pdf"  # place pdf here
    output_path = "data/processed/text_chunks.jsonl"

    extract_text_from_pdf(pdf_path, output_path)
