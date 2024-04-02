import xml.etree.ElementTree as ET
import re

class DataPostProcess:
    def __init__(self, input_data, prediction):
        self.input_data = input_data
        self.prediction = prediction
        
    @staticmethod
    def extract_and_remove_elements_by_tag(root, tag_name, parent_tag_name=None):
        parent = root.find(f'.//{parent_tag_name}')
        if parent is None: 
            return []
        elements = parent.findall(f'.//{tag_name}')
        for element in elements:
            parent.remove(element)
    
        return elements
    
    def data_postprocess(self):
        tree1 = ET.ElementTree(ET.fromstring(self.input_data))
        root1 = tree1.getroot()
        tree2 = ET.ElementTree(ET.fromstring(self.prediction))
        root2 = tree2.getroot()
        
        global_actions = self.extract_and_remove_elements_by_tag(root1, 'GlobalAction', 'Actions')
        stories = self.extract_and_remove_elements_by_tag(root1, 'Story', 'Storyboard')
        stop_triggers = self.extract_and_remove_elements_by_tag(root1, 'StopTrigger', 'Storyboard')

        actions = root2.find('.//Storyboard/Init/Actions')
        storyboard = root2.find('.//Storyboard')
        init_block = root2.find('.//Storyboard/Init')

        for i, action in enumerate(global_actions):
            actions.insert(i, action)

        for story in stories:
            init_index = list(storyboard).index(init_block) + 1
            storyboard.insert(init_index, story)

        last_story_index = [i for i, child in enumerate(storyboard) if child.tag == 'Story'][-1]
        for stop_trigger in stop_triggers:
            storyboard.insert(last_story_index + 1, stop_trigger)
            last_story_index += 1  
            
        pattern = re.compile(r'\s+(?=/>)')
        root = pattern.sub('', ET.tostring(root2, encoding='unicode', method='xml'))
        declaration = '<?xml version="1.0" ?>\n'
        root = declaration + root
        
        return root
    
# if __name__ == '__main__':
#     input_file = 'test_input.xml'
#     inter_file = 'test_inter.xml'
    
#     with open(input_file, 'r', encoding='utf-8') as input_file:
#         input_data = input_file.read()

#     with open(inter_file, 'r', encoding='utf-8') as input_file:
#         pred = input_file.read()
        
#     postprocess = DataPostProcess(input_data, pred)
#     output = postprocess.data_postprocess()
#     print(output)