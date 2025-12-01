import json
import os

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_combined_chunks(text_path, tables_path, images_path, output_path):
    chunks = []

    # Text chunks
    text_chunks = load_jsonl(text_path)
    for i, item in enumerate(text_chunks):
        chunks.append({
            "id": f"text_p{item['page']}_{i}",
            "type": "text",
            "page": item["page"],
            "content": item["content"]
        })

    # Table chunks
    tables = []
    if os.path.exists(os.path.join(tables_path, "tables.json")):
        with open(os.path.join(tables_path, "tables.json"), "r", encoding="utf-8") as f:
            tables = json.load(f)

    for i, table in enumerate(tables):
        chunks.append({
            "id": f"table_{i+1}",
            "type": "table",
            "page": table.get("page", None),
            "content": table["summary"]
        })

    # Image OCR chunks
    images = []
    if os.path.exists(images_path):
        with open(images_path, "r", encoding="utf-8") as f:
            images = json.load(f)

    for i, img in enumerate(images):
        chunks.append({
            "id": f"img_p{img['page']}_{i+1}",
            "type": "image",
            "page": img["page"],
            "content": img["ocr"]
        })

    # Save final output
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Chunk building complete. Total chunks: {len(chunks)}")


if __name__ == "__main__":
    build_combined_chunks(
        text_path="data/processed/text_chunks.jsonl",
        tables_path="data/processed/tables",
        images_path="data/processed/image_ocr.json",
        output_path="data/processed/combined_chunks.jsonl"
    )
