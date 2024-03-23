import carla
import random
from time import sleep
import xml.etree.ElementTree as ET
import logging
import subprocess
import os.path as path
from tqdm import tqdm

# Scenario runner root
SCENARIO_RUNNER_ROOT = path.expanduser("~") + "/MSU/LLM/life2scenario_core/dataset_generation/scenario_runner"

from scene_manipulator import Action, ActionEntityType, ActionType, ActionStrategy, Condition, SceneManipulator, get_role_name
from scenario_extender import ScenarioExtender, TrainData, NO_OP_PROMPTS

def get_current_scenario():
    try:
        # Query scenario_runner.py process
        process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
        output, error = process.communicate()
        output = output.decode()
        if 'scenario_runner.py' in output:
            # Get the scenario_runner.py arguments
            args = output.split(' ')
            # Get the scenario name
            scenario_name = args[args.index('--openscenario')+1].split("\n")[0]   
            return scenario_name
        else:
            return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None
# Check if scenario runner is running
def is_scenario_runner_running():
    try:
        # Query scenario_runner.py process
        process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
        output, error = process.communicate()
        output = output.decode()

        if 'scenario_runner.py' in output:
            return True
        else:
            return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def add_random_close_pedestrian(world: carla.World, cur_scene: TrainData, manip: SceneManipulator, scene_extender: ScenarioExtender):
    
    # Get all the vehicle actors
    vehicle_actors = world.get_actors().filter('vehicle.*')
    
    cur_scene_name = cur_scene.name
    cur_scene_tree = cur_scene.target

    random_vehicle = random.choice(vehicle_actors)
    random_vehicle_role = get_role_name(random_vehicle)
    
    res = manip.add_pedestrian_close_to(random_vehicle_role)
    if(res is not None):
        action, _ = res
        scene_extender.inform_action(action, cur_scene_tree, cur_scene_name)
        scene_extender.save_data_w_idx(len(scene_extender.dataset)-1)

def add_random_pedestrian(world: carla.World, cur_scene: TrainData, manip: SceneManipulator, scene_extender: ScenarioExtender):
    
    cur_scene_name = cur_scene.name
    cur_scene_tree = cur_scene.target
    
    random_location = carla.Location(x=random.uniform(-100, 100), y=random.uniform(-100, 100), z=0.5)
    random_transform = carla.Transform(random_location)
    res = manip.add_pedestrian(random_transform)
    if(res is not None):
        action, _ = res
        scene_extender.inform_action(action, cur_scene_tree, cur_scene_name)
        scene_extender.save_data_w_idx(len(scene_extender.dataset)-1)


def remove_random_pedestrian(world: carla.World, cur_scene: TrainData, manip: SceneManipulator, scene_extender: ScenarioExtender):
        
    cur_scene_name = cur_scene.name
    cur_scene_tree = cur_scene.target
    cur_ped_to_remove = random.choice(world.get_actors().filter('walker.pedestrian.*'))
    cur_ped_name_to_remove = get_role_name(cur_ped_to_remove)
    
    action = manip.remove_pedestrian(cur_ped_name_to_remove)
    if(action is not None):
        scene_extender.inform_action(action, cur_scene_tree, cur_scene_name)
        scene_extender.save_data_w_idx(len(scene_extender.dataset)-1)
        
def remove_random_close_pedestrian(world: carla.World, cur_scene: TrainData, manip: SceneManipulator, scene_extender: ScenarioExtender):
        
    # Get all the vehicle actors
    vehicle_actors = world.get_actors().filter('vehicle.*')
    
    cur_scene_name = cur_scene.name
    cur_scene_tree = cur_scene.target

    random_vehicle = random.choice(vehicle_actors)
    random_vehicle_role = get_role_name(random_vehicle)
    
    action = manip.remove_pedestrian_randomly_close_to(random_vehicle_role)
        
    if(action is not None):
        scene_extender.inform_action(action, cur_scene_tree, cur_scene_name)
        scene_extender.save_data_w_idx(len(scene_extender.dataset)-1)

def main():
    # Connect to the client and retrieve the world object 
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    print(f"Connected to {world.get_map().name} map")
    
    
    # Print warning
    while not is_scenario_runner_running():
        print("Scenario runner is not running, please start it first...")
    
    scenario_relative_path = get_current_scenario()
    print(f"Current scenario: {scenario_relative_path}")
    
    orig_scenario_full_path = path.join(SCENARIO_RUNNER_ROOT, scenario_relative_path)
    orig_scenario_name = path.splitext(path.basename(orig_scenario_full_path))[0]
    orig_tree = ET.parse(orig_scenario_full_path)
    orig_scene = TrainData(orig_tree, random.choice(NO_OP_PROMPTS), orig_tree, -1, orig_scenario_name)

    # Get all the vehicle actors
    vehicle_actors = world.get_actors().filter('vehicle.*')
    
    # Create a SceneManipulator object
    manip = SceneManipulator(world, orig_scenario_full_path)
    scene_extender = ScenarioExtender(world)
    
    
    # Randomly change the location    
    for session_idx in tqdm(range(10000)):
        
        # Remove all pedestrians
        for actor in world.get_actors().filter('walker.pedestrian.*'):
            actor.destroy()

        
        num_add_peds = random.randint(2, 30)
        num_remove_peds = random.randint(0, num_add_peds)
        
        
        for add_idx in range(num_add_peds):
            if(add_idx == 0):
                add_random_close_pedestrian(world, orig_scene, manip, scene_extender)
            else:
                random_add_function = random.choice([add_random_pedestrian, add_random_close_pedestrian])
                random_add_function(world, scene_extender.dataset[-1], manip, scene_extender)

        for _ in range(num_remove_peds):
            
            ped_actors = world.get_actors().filter('walker.pedestrian.*')
            if(len(ped_actors) == 0):
                break
            
            random_remove_function = random.choice([remove_random_pedestrian, remove_random_close_pedestrian])
            random_remove_function(world, scene_extender.dataset[-1], manip, scene_extender)
        
        
        print(f"Session {session_idx} done, Added/Removed {num_add_peds}/{num_remove_peds} pedestrians")
        # sleep(0.1)


    pass
if __name__ == "__main__":
    main()