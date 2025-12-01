import camelot
import os
import json

def extract_tables(pdf_path, output_folder):
    tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")  # or 'stream' if lattice misses
    os.makedirs(output_folder, exist_ok=True)

    results = []

    for i, table in enumerate(tables):
        csv_path = os.path.join(output_folder, f"table_{i+1}.csv")
        table.to_csv(csv_path)

        summary_text = " | ".join(table.df.iloc[0].tolist())
        chunk = {
            "table_id": i + 1,
            "summary": summary_text,
            "path": csv_path,
            "type": "table"
        }
        results.append(chunk)

    with open(os.path.join(output_folder, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Extracted {len(tables)} tables.")
    return results


if __name__ == "__main__":
    pdf_path = "data/raw/qatar_test_doc.pdf"
    output_folder = "data/processed/tables"

    extract_tables(pdf_path, output_folder)
