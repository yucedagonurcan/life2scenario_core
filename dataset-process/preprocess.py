import xml.etree.ElementTree as ET
import re

class DataPreprocess:
    def __init__(self, input_data):
        self.input_data = input_data
        
    @staticmethod    
    def remove_elements_by_tag(tag_name, root):
        for element in root.findall(f'.//{tag_name}'):
            parent = root.find(f'.//{tag_name}/..')
            if parent is not None:
                parent.remove(element)
        
    def convert_to_root(self):
        tree = ET.ElementTree(ET.fromstring(self.input_data))
        root = tree.getroot()
        self.remove_elements_by_tag('GlobalAction', root)
        self.remove_elements_by_tag('Story', root)
        self.remove_elements_by_tag('StopTrigger', root)
        
        pattern = re.compile(r'\s+(?=/>)')
        root = pattern.sub('', ET.tostring(root, encoding='unicode', method='xml'))
        declaration = '<?xml version="1.0" ?>\n'
        root = declaration + root
        
        return root
    
    def data_preprocess(self):
        root = self.convert_to_root()
        return root

if __name__ == '__main__':
    input_file = 'test_input.xml'

    with open(input_file, 'r', encoding='utf-8') as file:
        xml_data = file.read()

    preprocess = DataPreprocess(xml_data)
    output = preprocess.data_preprocess()
    print(output)
    
  
