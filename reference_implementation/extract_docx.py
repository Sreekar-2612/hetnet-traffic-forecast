import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(path):
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = ET.fromstring(xml_content)
    
    # Namespaces are important in docx XML
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    paragraphs = []
    for paragraph in tree.findall('.//w:p', ns):
        texts = [node.text for node in paragraph.findall('.//w:t', ns) if node.text]
        if texts:
            paragraphs.append("".join(texts))
    
    return "\n".join(paragraphs)

if __name__ == "__main__":
    content = extract_text_from_docx('TASTF_Implementation_Plan.docx')
    with open('plan_content.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Content extracted successfully to plan_content.txt")
