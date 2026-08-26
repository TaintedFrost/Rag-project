import pymupdf

pdf_path = "data/Regulament-Burse_2023-2024.pdf"

document = pymupdf.open(pdf_path)

print(f"Number of pages: {len(document)}")

for page_number, page in enumerate(document):
    text = page.get_text()

    print(f"\n--- PAGE {page_number + 1} ---\n")
    print(text)