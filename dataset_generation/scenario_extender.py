import carla
import random
from time import sleep
import xml.etree.ElementTree as ET
import logging
import subprocess
import os.path as path
import os
from scene_manipulator import Action, ActionEntityType, ActionType, ActionStrategy, Condition, SceneManipulator, get_role_name
import copy
import shutil
from xml.dom import minidom
from typing import List


PREFIX_PHRASES = ["Please", "Can you", "Could you", "Would you", "I need you to", "I want you to", 
                  "I would like you to", "I would like to ask you to", "I would like to request you to",
                  "I would like to tell you to", "I would like to instruct you to", "I would like to command you to",
                  "I would like to order you to", "I would like to direct you to", "I would like to guide you to",
                  "I would like to lead you to", "I would like you to demonstrate", "I would like you to display",
                  ""]

NO_OP_PROMPTS = ["Do nothing", "Do not do anything", "Do not do anything at all"]

class TrainData:
    
    def __init__(self, target: ET.Element, prompt: str, reference: ET.Element, in_id: int = 0, scenario_name: str = ""):
    
        self.target = target
        self.prompt = prompt
        self.reference = reference
        self.id = in_id
        self.name = scenario_name
        
# Create a ScenarioExtender class
class ScenarioExtender:
    
    # Constructor
    def __init__(self, world):
        
        self.world = world    
        self.blueprint_library = world.get_blueprint_library()
        
        self.dataset: List[TrainData] = []
        
        self.prepare_dataset()
        
    def prepare_dataset(self):
        
        # Get current file path
        current_file_path = path.abspath(__file__)
        
        self.scenario_save_root = path.join(path.dirname(current_file_path), "dataset")
        self.train_data_folder = path.join(self.scenario_save_root, "train")
        
        self.prompts_data_folder = path.join(self.train_data_folder, "prompts")
        self.reference_scenarios_folder = path.join(self.train_data_folder, "ref_scenarios")
        self.target_scenarios_folder = path.join(self.train_data_folder, "target_scenarios")
        
        if not path.exists(self.scenario_save_root):
            os.makedirs(self.scenario_save_root)
        
        if not path.exists(self.train_data_folder):
            os.makedirs(self.train_data_folder)
            
        if not path.exists(self.prompts_data_folder):
            os.makedirs(self.prompts_data_folder)
        
        if not path.exists(self.reference_scenarios_folder):
            os.makedirs(self.reference_scenarios_folder)
        
        if not path.exists(self.target_scenarios_folder):
            os.makedirs(self.target_scenarios_folder)
            
        
    def reflect_xml_pedestrian_remove(self, action: Action, tree: ET.ElementTree):
            
        # Creat a temp root
        tmp_tree = copy.deepcopy(tree)
        tmp_root = tmp_tree.getroot()
        
        entities = tmp_root.find('Entities')
        entity_to_remove = entities.find(f"ScenarioObject[@name='{action.self_entity_name}']")
        entities.remove(entity_to_remove)
        
        actions = tmp_root.find('Storyboard/Init/Actions')
        action_to_remove = actions.find(f"Private/[@entityRef='{action.self_entity_name}']")
        actions.remove(action_to_remove)
        
        return tmp_tree
    
    def reflect_xml_pedestrian_add(self, action: Action, tree: ET.ElementTree):
        
        # Creat a temp root
        tmp_tree = copy.deepcopy(tree)
        tmp_root = tmp_tree.getroot()
        
        entities = tmp_root.find('Entities')

        pedestrian_name = f"{action.self_entity_name}"
        
        entity = ET.Element('ScenarioObject')
        entity.set('name', pedestrian_name)
        
        # Add Pedestrian
        pedestrian = ET.SubElement(entity, 'Pedestrian')
        pedestrian.set("model", "walker.pedestrian.0001")
        pedestrian.set("mass", "80")
        pedestrian.set("name", "walker.pedestrian.0001")
        pedestrian.set("pedestrianCategory", "pedestrian")
        
        # Add ParameterDeclarations
        param_decl = ET.SubElement(pedestrian, 'ParameterDeclarations')

        # Add Bounding Box
        bbox = ET.SubElement(pedestrian, 'BoundingBox')
        
        # Add Center
        center = ET.SubElement(bbox, 'Center')
        center.set('x', str(1.5))
        center.set('y', str(0.0))
        center.set('z', str(0.9))
        
        # Add Dimension
        dimension = ET.SubElement(bbox, 'Dimensions')
        dimension.set('width', '2')
        dimension.set('length', '2')
        dimension.set('height', '2')
        
        # Add Properties
        properties = ET.SubElement(pedestrian, 'Properties')
        
        # Add Property 
        prop = ET.Element('Property')
        
        prop.set('name', 'type')
        prop.set('value', 'simulation')
        
        properties.append(prop)
        
        entities.append(entity)

        # Add Entity
        private = ET.Element('Private')
        private.set('entityRef', pedestrian_name)
        
        # Get PrivateAction
        private_action = ET.SubElement(private, 'PrivateAction')
        # Get TeleportAction
        teleport_action = ET.SubElement(private_action, 'TeleportAction')
        
        # Get Position
        position = ET.SubElement(teleport_action, 'Position')
        
        # Get WorldPosition
        world_position = ET.SubElement(position, 'WorldPosition')
        world_position.set('x', str(action.location.location.x))
        world_position.set('y', str(action.location.location.y))
        world_position.set('z', str(action.location.location.z))
        
            
        # Get Storyboard
        actions = tmp_root.find('Storyboard/Init/Actions')
        
        actions.append(private)
        
        return tmp_tree
    
    def export_train_data(self):

        pretty_print_xml = lambda content: '\n'.join(
            [line for line in minidom.parseString(content).toprettyxml(indent=' '*2).split('\n') if line.strip()])
        
        for t_idx in range(len(self.targets)):

            orig_scenario_name = path.splitext(path.basename(self.scenario_path))[0]
            current_scenario_suffix = str(self.action_list[t_idx].index).zfill(7)
            
            out_scenario_name = orig_scenario_name + "_" + current_scenario_suffix
            
            # Save the target
            target_filename = out_scenario_name + f".xosc"
            target_save_path = path.join(self.target_scenarios_folder, target_filename)
            
            xmlstr = pretty_print_xml(ET.tostring(self.targets[t_idx].getroot()))
            with open(target_save_path, "w") as f:
                f.write(xmlstr)
            # self.targets[t_idx].write(target_save_path)
            
            
            # Save the prompts
            prompt_filename = out_scenario_name + f".txt"
            prompt_save_path = path.join(self.prompts_data_folder, prompt_filename)
            with open(prompt_save_path, "w") as f:
                f.write(self.prompts[t_idx])

            # Copy the original scenario
            ref_filename = out_scenario_name + f".xosc"
            ref_save_path = path.join(self.reference_scenarios_folder, ref_filename)
            shutil.copy(self.scenario_path, ref_save_path)
            
            
    def reflect_prompt_pedestrian_add(self, action: Action):
        
        # lambda function for floation point rounding
        round_float = lambda x: round(x, 2)
        
        # lambda function for location string
        location_str = lambda loc: f"({round_float(loc.location.x)}, {round_float(loc.location.y)}, {round_float(loc.location.z)})"

        selected_prefix = random.choice(PREFIX_PHRASES)
        
        if action.action_strategy == ActionStrategy.ADD_W_LOCATION:
            return f"{selected_prefix} add pedestrian at location {location_str(action.location)}".strip().lower()
        elif action.action_strategy == ActionStrategy.ADD_CLOSE_TO:
            return f"{selected_prefix} add pedestrian close to {action.related_entity_name}".strip().lower()
        else:
            return f"{selected_prefix} add pedestrian".strip().lower()
        
    def reflect_prompt_pedestrian_remove(self, action: Action):

        selected_prefix = random.choice(PREFIX_PHRASES)
        
        if action.action_strategy == ActionStrategy.REMOVE_W_NAME:
            return f"{selected_prefix} remove pedestrian actor named {action.self_entity_name}".strip().lower()
        elif action.action_strategy == ActionStrategy.REMOVE_CLOSE_TO:
            return f"{selected_prefix} remove pedestrian close to {action.related_entity_name}".strip().lower()
        else:
            return f"{selected_prefix} remove pedestrian".strip().lower()
    
    def inform_action(self, action: Action, orig_tree: ET.ElementTree, orig_scenario_name: str) -> bool:
        
        
        if action.entity_type == ActionEntityType.PEDESTRIAN and action.action_type == ActionType.ADD:
            reference = orig_tree
            target = self.reflect_xml_pedestrian_add(action, orig_tree)
            prompt = self.reflect_prompt_pedestrian_add(action)
            
            
            self.dataset.append(TrainData(target, prompt, reference, action.index, orig_scenario_name))
            return True
        if action.entity_type == ActionEntityType.PEDESTRIAN and action.action_type == ActionType.REMOVE:
            reference = orig_tree
            target = self.reflect_xml_pedestrian_remove(action, orig_tree)
            prompt = self.reflect_prompt_pedestrian_remove(action)
            
            self.dataset.append(TrainData(target, prompt, reference, action.index, orig_scenario_name))
            return True
    
    def pretty_print_xml(self, content) -> str:
        return '\n'.join(
            [line for line in minidom.parseString(content).toprettyxml(indent=' '*2).split('\n') if line.strip()])
    
    def save_xml(self, tree: ET.ElementTree, save_path: str) -> None:
        
        xmlstr = self.pretty_print_xml(ET.tostring(tree.getroot()))
        with open(save_path, "w") as f:
            f.write(xmlstr)
            
    def save_prompt(self, prompt: str, save_path: str) -> None:
        with open(save_path, "w") as f:
            f.write(prompt)
    
    def save_data_w_idx(self, idx: int) -> bool:
        
        data = self.dataset[idx]
        
        scene_target = data.target
        scene_prompt = data.prompt
        scene_reference = data.reference
        scene_name = data.name
        scene_id = data.id

        current_scenario_suffix = str(scene_id).zfill(7)
        out_scenario_name = scene_name + "_" + current_scenario_suffix

        #? Save the target #########################################
        target_filename = out_scenario_name + f".xosc"
        target_save_path = path.join(self.target_scenarios_folder, target_filename)
        self.save_xml(scene_target, target_save_path)
        ############################################################

        #! Save the prompts #########################################
        prompt_filename = out_scenario_name + f".txt"
        prompt_save_path = path.join(self.prompts_data_folder, prompt_filename)
        self.save_prompt(scene_prompt, prompt_save_path)
        ############################################################

        #? Copy the original scenario ###############################
        ref_filename = out_scenario_name + f".xosc"
        ref_save_path = path.join(self.reference_scenarios_folder, ref_filename)
        self.save_xml(scene_reference, ref_save_path)
        ############################################################
        
        print(f"Saved data for scenario {scene_prompt}: with id {scene_id} to {out_scenario_name}")
        
        return True
        
    def return_last_informed(self):
        return self.dataset[-1]
