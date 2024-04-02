import os
import re
from tqdm import tqdm   
    
class CountMetric:
    def __init__(self, prompts, references, ground_truths, predictions):
        self.prompts = prompts
        self.references = references
        self.ground_truths = ground_truths
        self.predictions = predictions
        
    @staticmethod
    def find_verb(prompt):
        verbs = ['add', 'remove']
        for verb in verbs:
            if verb.lower() in prompt.lower():
                return verb.lower()
        return None
    
    @staticmethod
    def count_object(code, target_word):
        pattern = r'\b' + re.escape(target_word) + r'\b'
        matches = re.findall(pattern, code, re.IGNORECASE)
        return len(matches)
    
    @staticmethod
    def is_count_match(prompt, reference, ground_truth, prediction):
        verb = CountMetric.find_verb(prompt)
        ref_count = CountMetric.count_object(reference, verb)
        gt_count = CountMetric.count_object(ground_truth, verb)
        pred_count = CountMetric.count_object(prediction, verb)
        gt_check = 0
        pred_check = 0
        if verb == 'add':
            if ref_count + 1 == gt_count:
                gt_check = 1
            if ref_count + 1 == pred_count:
                pred_check = 1
        elif verb == 'remove':
            if ref_count - 1 == gt_count:
                gt_check = 1
            if ref_count - 1 == pred_count:
                pred_check = 1
        return ~(gt_check ^ pred_check) + 2
    
    def calculate_accuracy(self):
        cnt = 0
        length = len(self.ground_truths)
        for i in tqdm(range(length), desc="Calculating count"):
            cnt += self.is_count_match(self.prompts[i], self.references[i], self.ground_truths[i], self.predictions[i])
        return (cnt / length) * 100

# if __name__ == '__main__':
#     prompts = []
#     references = []
#     ground_truths = []

#     prompt_directory = '../../dataset/train/prompts'
#     filenames = [f for f in os.listdir(prompt_directory) if f.endswith('.txt')]
#     i = 0
#     for filename in tqdm(filenames, desc="Reading files"):
#         filepath = os.path.join(prompt_directory, filename)
#         with open(filepath, 'r', encoding='utf-8') as file:
#             prompts.append(file.read())
#         i += 1
#         if i == 2000:
#             break
    
#     ref_directory = '../../dataset/train/ref_scenarios'
#     filenames = [f for f in os.listdir(ref_directory) if f.endswith('.xosc')]
#     i = 0
#     for filename in tqdm(filenames, desc="Reading files"):
#         filepath = os.path.join(ref_directory, filename)
#         with open(filepath, 'r', encoding='utf-8') as file:
#             references.append(file.read())
#         i += 1
#         if i == 2000:
#             break
        
#     gt_directory = '../../dataset/train/target_scenarios'
#     filenames = [f for f in os.listdir(gt_directory) if f.endswith('.xosc')]
#     i = 0
#     for filename in tqdm(filenames, desc="Reading files"):
#         filepath = os.path.join(gt_directory, filename)
#         with open(filepath, 'r', encoding='utf-8') as file:
#             ground_truths.append(file.read())
#         i += 1
#         if i == 2000:
#             break
    

#     checker = CountMetric(prompts, references, ground_truths, ground_truths)
#     accuracy = checker.calculate_accuracy()
#     print(f'Accuracy: {accuracy}%')
