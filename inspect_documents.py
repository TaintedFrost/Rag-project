import pymupdf
from pathlib import Path

data_folder = Path("data")

pdf_files = list(data_folder.glob("*.pdf"))

for pdf_path in pdf_files:
    document = pymupdf.open(pdf_path)

    first_page = document[0].get_text()

    print("=" * 70)
    print(f"FILE: {pdf_path.name}")
    print(f"PAGES: {len(document)}")
    print("=" * 70)
    print(first_page[:1500])
    print()

    document.close()