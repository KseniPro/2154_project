import argparse
import os
import pypdfium2 as pdfium


def convert_pdf_to_images(pdf_path, output_dir, image_format="png", dpi=300, page=None):
    """
    Converts each page or a specific page of a PDF to images using pypdfium2.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Path to the folder where images will be saved.
        image_format (str): Image format (e.g., "png", "jpeg").
        dpi (int): Resolution in dots per inch.
        page (int, optional): Specific page to convert (1-based index). If None, converts all pages.
    """
    pdf = pdfium.PdfDocument(pdf_path)

    if page is not None:
        if page < 1 or page > len(pdf):
            raise ValueError(f"Page number {page} is out of range. The PDF has {len(pdf)} pages.")
        page_index = page - 1
        page_obj = pdf.get_page(page_index)
        pil_image = page_obj.render(scale=dpi / 72).to_pil()
        output_filename = f"page_{page}.{image_format}"
        output_path = os.path.join(output_dir, output_filename)
        pil_image.save(output_path)
    else:
        for page_index in range(len(pdf)):
            page_obj = pdf.get_page(page_index)
            pil_image = page_obj.render(scale=dpi / 72).to_pil()
            output_filename = f"page_{page_index + 1}.{image_format}"
            output_path = os.path.join(output_dir, output_filename)
            pil_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Convert PDF pages to images.')
    parser.add_argument('input_pdf', help='Path to the input PDF file')
    parser.add_argument('output_dir', help='Path to the output directory for images')
    parser.add_argument('--image-format', default='png',
                        help='Image format (e.g., png, jpeg) (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Resolution in DPI (default: 300)')
    parser.add_argument('--page', type=int,
                        help='Convert only a specific page (1-based index)')
    args = parser.parse_args()

    if not os.path.exists(args.input_pdf):
        raise FileNotFoundError(f"Input PDF file '{args.input_pdf}' does not exist.")

    os.makedirs(args.output_dir, exist_ok=True)

    convert_pdf_to_images(args.input_pdf, args.output_dir, args.image_format, args.dpi, args.page)

    if args.page:
        print(f"Successfully converted page {args.page} to {args.image_format} in '{args.output_dir}'")
    else:
        print(f"Successfully converted all PDF pages to {args.image_format} images in '{args.output_dir}'")

if __name__ == "__main__":
    main()
