import zipfile
import xml.etree.ElementTree as ET

docx_path = r"c:\Users\ryoar\Desktop\infosys project\Autonomous Cognitive Engine for Deep.docx"

try:
    with zipfile.ZipFile(docx_path) as docx:
        # Check files in zip
        print("Files in docx:")
        for name in docx.namelist():
            print(f"- {name}")
            
        content = docx.read('word/document.xml')
        root = ET.fromstring(content)
        
        print(f"\nRoot tag: {root.tag}")
        
        # Print namespace map if possible, elementtree doesn't expose it directly but tags have it
        # Just iterate first few elements
        print("\nFirst 10 elements:")
        for i, elem in enumerate(root.iter()):
            print(f"{i}: {elem.tag} text='{elem.text}'")
            if i >= 10: break
            
        # Try to extract all text regardless of structure
        all_text = []
        for elem in root.iter():
            if elem.text:
                all_text.append(elem.text)
        
        with open('debug_text.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
            
        print(f"\nExtracted {len(all_text)} text fragments to debug_text.txt")

except Exception as e:
    print(f"Error: {e}")
