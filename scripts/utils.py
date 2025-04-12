import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image

def create_image_folder(folder_name="images"):
    """Create directory for storing visualization images"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created visualization folder: {folder_name}")
    else:
        print(f"Visualization folder {folder_name} already exists")

def collate_images_to_pdf(image_folder="images", output_pdf="EDA_Report.pdf"):
    """Combine all generated images into a PDF report"""
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    if not image_files:
        print("No images found for PDF creation")
        return

    pdf_path = os.path.join(image_folder, output_pdf)
    c = canvas.Canvas(pdf_path, pagesize=A4)

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                aspect = img_height / img_width

                # Calculate dimensions to fit A4
                max_width = A4[0] - 100
                max_height = A4[1] - 100
                display_width = min(max_width, img_width)
                display_height = display_width * aspect

                if display_height > max_height:
                    display_height = max_height
                    display_width = display_height / aspect

                x = (A4[0] - display_width) / 2
                y = (A4[1] - display_height) / 2

                c.drawImage(img_path, x, y, width=display_width, height=display_height)
                c.showPage()
        except Exception as e:
            print(f"Error adding {img_file} to PDF: {str(e)}")

    c.save()
    print(f"PDF report saved to: {pdf_path}")