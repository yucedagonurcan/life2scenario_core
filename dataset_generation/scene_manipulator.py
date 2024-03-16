import carla
import random
from time import sleep
import xml.etree.ElementTree as ET
import logging
import subprocess
import os.path as path


def generate_random_relative_transform(entity_transform: carla.Transform, min_distance: float, max_distance: float):
    # Generate random relative transform
    relative_transform = entity_transform
    relative_transform.location = carla.Location(   entity_transform.location.x + random.uniform(min_distance, max_distance),
                                                    entity_transform.location.y + random.uniform(min_distance, max_distance),
                                                    entity_transform.location.z)
    return relative_transform



def find_actor_by_role_name(world: carla.World, role_name: str):
    
    # Get all actors
    actors = world.get_actors()
    
    # Find the actor with the given role name
    for actor in actors:
        # Check if it has the attribute
        if "role_name" in actor.attributes:
                # Check if the role name matches
                if actor.attributes['role_name'] == role_name:
                    return actor
    return None

def get_role_name(actor: carla.Actor):
    
    if "role_name" in actor.attributes:
        return actor.attributes['role_name']
    else:
        return None

class ActionType:
    
    ADD = 0
    REMOVE = 1
    MODIFY = 2

class ActionStrategy:
    
    ADD_CLOSE_TO = 0
    ADD_W_LOCATION = 1

    REMOVE_RANDOM = 2
    REMOVE_CLOSE_TO = 3
    REMOVE_W_NAME = 4

    MODIFY_EXTEND = 5
    MODIFY_SHRINK = 6

    NO_OP = 6

class ActionEntityType:
    
    VEHICLE = 0
    PEDESTRIAN = 1
    RANDOM = 2


    
class Action:
    def __init__(self, index: int, action_type, entity_type, action_strategy, location=None, actor: carla.Actor = None,
                 self_entity_name=None, related_entity_name=None):
        self.action_type = action_type
        self.action_strategy = action_strategy
        self.entity_type = entity_type
        self.location = location
        self.self_entity_name = self_entity_name
        self.related_entity_name = related_entity_name
        self.actor = actor
        
        # Create an index for the action
        self.index = index

class Condition:
    def __init__(self, condition_type, condition_value):
        self.condition_type = condition_type
        self.condition_value = condition_value


class SceneManipulator:

    # Constructor
    def __init__(self, world: carla.World, scenario_path: str):
        
        self.world = world
        self.action_idx = 0
        
        self.scenario_path = scenario_path
        self.tree = ET.parse(scenario_path)
        self.root = self.tree.getroot()
        
        self.num_entities = len(self.root.find('Entities'))
        self.blueprint_library = world.get_blueprint_library()
        
        self.pedestrian_bp = self.blueprint_library.filter("walker.pedestrian.*")
        self.vehicle_bp = self.blueprint_library.filter("vehicle.*")
        

        
    # Add a pedestrian to the scenario
    def add_pedestrian(self, location: carla.Location):
        
        self.action_idx += 1
        
        # Select random pedestrian from blueprint library
        selected_ped_bp = random.choice(self.pedestrian_bp)
        role_name = f"pedestrian_w_transform_{self.action_idx}"

        selected_ped_bp.set_attribute('role_name', role_name)

        # Spawn the pedestrian
        try:
            pedestrian = self.world.spawn_actor(selected_ped_bp, location)
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        
        # Set the pedestrian to walk
        pedestrian.set_simulate_physics(True)

        return Action(self.action_idx, ActionType.ADD, ActionEntityType.PEDESTRIAN, ActionStrategy.ADD_W_LOCATION, location=location, actor=pedestrian, self_entity_name=role_name, related_entity_name=None), pedestrian
    
    
    def add_pedestrian_close_to(self, relative_entity_name: str) -> (Action, carla.Actor):

        self.action_idx += 1
        
        # Select random pedestrian from blueprint library
        selected_ped_bp = random.choice(self.pedestrian_bp)
        
        # Get the relative entity
        relative_entity = find_actor_by_role_name(self.world, relative_entity_name)
        
        if relative_entity is None:
            return None
        
        # Get the relative entity location
        relative_entity_transform = relative_entity.get_transform()
        
        final_spawn_transform = generate_random_relative_transform(relative_entity_transform, -10, 10)        
        role_name = f"pedestrian_close_to_{relative_entity_name}_{self.action_idx}"
        selected_ped_bp.set_attribute('role_name', role_name)
        
        while True:
        
            # Spawn the pedestrian
            try:
                pedestrian = self.world.spawn_actor(selected_ped_bp, final_spawn_transform)
            except Exception as e:
                print(f"Error: {e}")
                final_spawn_transform = generate_random_relative_transform(relative_entity_transform, -10, 10)
                continue        
            break
            
        # Set the pedestrian to walk
        pedestrian.set_simulate_physics(True)
                
        return Action(self.action_idx, ActionType.ADD, ActionEntityType.PEDESTRIAN, ActionStrategy.ADD_CLOSE_TO, final_spawn_transform, actor=pedestrian, self_entity_name=role_name, related_entity_name=relative_entity_name), pedestrian
    
    
    def remove_pedestrian(self, entity_name: str):
        self.action_idx += 1

        # Get the pedestrian
        pedestrian = find_actor_by_role_name(self.world, entity_name)
        
        if not pedestrian:
            return None
        
        if not pedestrian.destroy():
            return None
        return Action(self.action_idx, ActionType.REMOVE, ActionEntityType.PEDESTRIAN, ActionStrategy.REMOVE_W_NAME, self_entity_name=entity_name, related_entity_name=entity_name)
        
    def remove_pedestrian_randomly_close_to(self, entity_name: str):
        self.action_idx += 1

        # Get the pedestrian
        vehicle = find_actor_by_role_name(self.world, entity_name)
        
        if not vehicle:
            return None
        
        # Get all pedestrians
        pedestrians = self.world.get_actors().filter("walker.pedestrian.*")
        
        if len(pedestrians) == 0:
            return None
        
        for pedestrian in pedestrians:
            if pedestrian.get_location().distance(vehicle.get_location()) < 10:
                role_name = pedestrian.attributes["role_name"]
                pedestrian.destroy()
                return Action(self.action_idx, ActionType.REMOVE, ActionEntityType.PEDESTRIAN, ActionStrategy.REMOVE_CLOSE_TO, self_entity_name=role_name, related_entity_name=entity_name)
        
        return None
        
    def remove_pedestrian_randomly(self):

        self.action_idx += 1

        # Get all pedestrians
        pedestrians = self.world.get_actors().filter("walker.pedestrian.*")
        
        if len(pedestrians) == 0:
            return None
        
        # Select random pedestrian
        pedestrian = random.choice(pedestrians)
        
        # Get pedestrian object's name
        pedestrian_name = pedestrian.attributes["role_name"]
        
        if not pedestrian.destroy():        
            return None
        
        return Action(self.action_idx, ActionType.REMOVE, ActionEntityType.PEDESTRIAN, ActionStrategy.REMOVE_RANDOM, actor=pedestrian, self_entity_name=pedestrian_name, related_entity_name=None)
        