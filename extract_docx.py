import zipfile
import re
import xml.etree.ElementTree as ET

docx_path = r"c:\Users\ryoar\Desktop\infosys project\Autonomous Cognitive Engine for Deep.docx"

try:
    with zipfile.ZipFile(docx_path) as docx:
        content = docx.read('word/document.xml')
        root = ET.fromstring(content)
        
        # Define namespace explicitly
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for p in root.findall('.//w:p', ns):
            p_text = ''
            for t in p.findall('.//w:t', ns):
                if t.text:
                    p_text += t.text
        with open('extracted_text_utf8.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))
        print("Extraction complete.")

except Exception as e:
    with open('extracted_text_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(f"Error reading docx: {e}")
    print(f"Error: {e}")
