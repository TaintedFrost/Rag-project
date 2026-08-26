import pymupdf
from pathlib import Path

data_folder = Path("data")
output_folder = Path("extracted_text")

output_folder.mkdir(exist_ok=True)

pdf_files = list(data_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF file(s).\n")

for pdf_path in pdf_files:
    print(f"Processing: {pdf_path.name}")

    document = pymupdf.open(pdf_path)

    all_text = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        all_text.append(
            f"\n--- PAGE {page_number + 1} ---\n{text}"
        )

    document.close()

    output_path = output_folder / f"{pdf_path.stem}.txt"

    output_path.write_text(
        "\n".join(all_text),
        encoding="utf-8"
    )

    print(f"Saved: {output_path}")
    print(f"Pages: {len(all_text)}\n")

print("Done!")