import streamlit as st

def convert_to_pdf():
    st.title("Guide to Convert PowerPoint to PDF")

    # Display a header
    st.header("Convert PowerPoint to PDF Using iLovePDF")

    # Display a message with a clickable link
    st.write("""
        To convert a PowerPoint presentation (PPTX) to a PDF file, follow these steps:

        1. **Download the PowerPoint File**: Ensure you have your PowerPoint file downloaded.
        2. **Go to iLovePDF**: Open your web browser and go to the [iLovePDF PowerPoint to PDF Converter](https://www.ilovepdf.com/powerpoint_to_pdf).
        3. **Upload Your PPTX File**:
            - Click on the "Select PowerPoint presentation" button on the iLovePDF website.
            - Browse and select your PPTX file from your computer.
        4. **Convert the File**:
            - After uploading the file, click on the "Convert to PDF" button.
            - The website will process your file and convert it to a PDF format.
        5. **Download the PDF**:
            - Once the conversion is complete, you will see a download button for the PDF file.
            - Click on "Download PDF" to save the converted file to your computer.

        Follow these steps to easily convert your PowerPoint presentations into PDF files for sharing or printing.
    """)

