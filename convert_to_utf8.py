try:
    with open('extracted_text.txt', 'r', encoding='utf-16le') as f:
        content = f.read()
    with open('extracted_text_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Converted successfully.")
except Exception as e:
    print(f"Error converting: {e}")
