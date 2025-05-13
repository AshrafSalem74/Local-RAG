import os
from PyPDF2 import PdfReader
import docx
import tempfile

def extract_text_from_file(uploaded_file):
    _, file_extension = os.path.splitext(uploaded_file.name.lower())
    
    if file_extension == ".txt":
        return uploaded_file.read().decode("utf-8")

    elif file_extension == ".pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    elif file_extension == ".docx":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_extension in [".md", ".html", ".htm", ".csv"]:
        try:
            from unstructured.partition.auto import partition
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            elements = partition(filename=tmp_path)
            return "\n".join([e.text for e in elements if e.text])

        except ImportError:
            return "[unstructured not installed]"

    else:
        return f"❌ Unsupported file type: {file_extension}"
