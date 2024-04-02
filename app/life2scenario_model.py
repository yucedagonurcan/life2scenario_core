import carla
import subprocess
import random
import re


class EntityType:
    
    PEDESTRIAN = 0
    VEHICLE = 1
    UNKNOWN = 2

class Life2ScenarioTaskProcessor:
    def __init__(self):
        self.simulator = Simulator()
        
    
    def find_variables(self, text):
        pattern = r'k\w*_\w*'
        return re.findall(pattern, text)
    
    def process_task(self, cur_task):
        
        cur_task = cur_task.replace("- ", "")
        
        # Store all variables into `self`
        variable_list = self.find_variables(cur_task)
        for cur_var in variable_list:
            cur_task = cur_task.replace(cur_var, f"self.{cur_var}")
        
        cur_task_refined=cur_task
        if "=" in cur_task.split("(")[0]:
            cur_task_function_part = cur_task.split("=")[1]
            cur_task_refined = cur_task.replace(cur_task_function_part, f"self.{cur_task_function_part.strip()}")
        else:
            cur_task_refined = f"self.{cur_task.strip()}"

        print(f"Executing: {cur_task_refined}")
        exec(cur_task_refined)
    
    def process_task_list(self, tasklist):

        
        for cur_task in tasklist:
            self.process_task(cur_task)
            
                
    def ADD_PEDESTRIAN(self, location=None, related_entity=None, self_entity_name=None):
        print(f"Adding pedestrian at location {location} related to {related_entity} with name {self_entity_name}")
        
        if location is not None:
            self.simulator.add_object(location, EntityType.PEDESTRIAN, self_entity_name)
        elif related_entity is not None:
            self.simulator.add_object_close_to(related_entity, EntityType.PEDESTRIAN, self_entity_name)
        else:
            self.simulator.add_object(None, EntityType.PEDESTRIAN, self_entity_name)
        
        
    
    def GET_RANDOM_PEDESTRIAN(self, related_entity=None):
        print(f"Getting random pedestrian related to {related_entity}")
        
        return self.simulator.get_random_object(related_entity, EntityType.PEDESTRIAN)
    
    def GET_RANDOM_VEHICLE(self, related_entity=None):
        print(f"Getting random vehicle related to {related_entity}")
        
        return self.simulator.get_random_object(related_entity, EntityType.VEHICLE)

    def REMOVE_PEDESTRIAN(self, related_entity=None, self_entity=None):
        print(f"Removing pedestrian with name {self_entity} related to {related_entity}")
        
        if related_entity is not None:
            return self.simulator.remove_object_close_to(related_entity, EntityType.PEDESTRIAN, self_entity)
        else:
            return self.simulator.remove_object(self_entity)
        
    
    def REMOVE_VEHICLE(self, related_entity=None, self_entity=None):
        print(f"Removing vehicle with name {self_entity} related to {related_entity}")
        if related_entity is not None:
            return self.simulator.remove_object_close_to(related_entity, EntityType.VEHICLE, self_entity)
        else:
            return self.simulator.remove_object(self_entity)
            

class Simulator:
    def __init__(self):
        
        self.simulator_running = self.try_connect_simulator()
        self.scenario_running = self.is_scenario_runner_running()
        
        self.action_idx = 0
        self.connection_established = False
        
        if self.simulator_running and self.scenario_running:
            self.connect_simulator()
    
    def get_vehicle_actors(self):
        if not self.connection_established:
            return None
        
        vehicle_actors = self.world.get_actors().filter('vehicle.*')
        return vehicle_actors
    
    def get_pedestrian_actors(self):
        if not self.connection_established:
            return None
        
        pedestrian_actors = self.world.get_actors().filter('walker.pedestrian.*')
        return pedestrian_actors
    
    def connect_simulator(self):
        if not self.connection_established:
            try:
                self.client = carla.Client('localhost', 2000)
                self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
        
                self.pedestrian_bp = self.blueprint_library.filter("walker.pedestrian.*")
                self.vehicle_bp = self.blueprint_library.filter("vehicle.*")
                
                self.connection_established = True
                return True
            except Exception as e:
                print(f"Error: {e}")
                return False
        else:
            return True
            
    # Check if scenario runner is running
    def is_scenario_runner_running(self):
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
    
    def try_connect_simulator(self):
        try:
            client = carla.Client('localhost', 2000)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    
        
    def find_actor_by_role_name(self, role_name: str):
        
        # Get all actors
        actors = self.world.get_actors()
        
        # Find the actor with the given role name
        for actor in actors:
            # Check if it has the attribute
            if "role_name" in actor.attributes:
                    # Check if the role name matches
                    if actor.attributes['role_name'] == role_name:
                        return actor
        return None


    def generate_transform_from_location(self, location: tuple):
        
        transform = carla.Transform()
        transform.location = carla.Location(location[0], location[1], location[2])
        
        return transform
    
    def generate_random_relative_transform(self, entity_transform: carla.Transform, min_distance: float, max_distance: float):
        # Generate random relative transform
        relative_transform = entity_transform
        relative_transform.location = carla.Location(   entity_transform.location.x + random.uniform(min_distance, max_distance),
                                                        entity_transform.location.y + random.uniform(min_distance, max_distance),
                                                        entity_transform.location.z)
        return relative_transform

    def get_name(self, entity):
        entity_name = None
        if isinstance(entity, carla.Actor):
            entity_name = entity.attributes['role_name']
        else:
            entity_name = entity
        return entity_name
    
    def get_close_entities(self, relative_entity, reference_entity_type: EntityType):
        
        relative_entity_name = self.get_name(relative_entity)
        relative_entity = self.find_actor_by_role_name(relative_entity_name)
        
        close_objects = []
        objects_to_check = []
        
        if reference_entity_type == EntityType.UNKNOWN:
            objects_to_check.extend(self.get_pedestrian_actors())
            objects_to_check.extend(self.get_vehicle_actors())
        elif reference_entity_type == EntityType.PEDESTRIAN:
            objects_to_check = self.get_pedestrian_actors()
        elif reference_entity_type == EntityType.VEHICLE:
            objects_to_check = self.get_vehicle_actors()                                    
    
        for cur_object in objects_to_check:
            if cur_object.get_location().distance(relative_entity.get_location()) < 10:
                close_objects.append(cur_object)
                
        return close_objects


    def add_object(self, location, reference_entity_type: EntityType, self_entity_name=None):
        
        object_bp = None
        if reference_entity_type == EntityType.PEDESTRIAN:
            object_bp = random.choice(self.pedestrian_bp)
        elif reference_entity_type == EntityType.VEHICLE:
            object_bp = random.choice(self.vehicle_bp)
    
        role_name = ""

        if self_entity_name is None:
            if reference_entity_type == EntityType.PEDESTRIAN:
                role_name = f"pedestrian_w_transform_{self.action_idx}"
            elif reference_entity_type == EntityType.VEHICLE:
                role_name = f"vehicle_w_transform_{self.action_idx}"
        else:
            role_name = self_entity_name
        

        self.action_idx += 1
        
        object_bp.set_attribute('role_name', role_name)

        
        spawn_transform = None
        
        if isinstance(location, tuple):
            spawn_transform = self.generate_transform_from_location(location)
        elif isinstance(location, carla.Transform):
            spawn_transform = location
        else:
            return None
        
        # Spawn the spawned_object
        try:
            spawned_object = self.world.spawn_actor(object_bp, spawn_transform)
            return spawned_object
        except Exception as e:
            print(f"Error: {e}")
            return None



    def add_object_close_to(self, relative_entity, reference_entity_type: EntityType, self_entity_name=None):

        relative_entity_name = self.get_name(relative_entity)
        
        # Get the relative entity
        relative_entity = self.find_actor_by_role_name(relative_entity_name)
        if relative_entity is None:
            return None
        
        # Get the relative entity location
        relative_entity_transform = relative_entity.get_transform()
        
        final_spawn_transform = self.generate_random_relative_transform(relative_entity_transform, -10, 10)    
        
        role_name = ""
        
        if self_entity_name is None:
            if reference_entity_type == EntityType.PEDESTRIAN:
                role_name = f"pedestrian_close_to_{relative_entity_name}_{self.action_idx}"
            elif reference_entity_type == EntityType.VEHICLE:
                role_name = f"vehicle_close_to_{relative_entity_name}_{self.action_idx}"
        else:
            role_name = self_entity_name
            
        
        self.action_idx += 1

        
        while True:
        
            # Spawn the spawned_object
            try:
                spawned_object = self.add_object(final_spawn_transform, reference_entity_type, role_name)
                if spawned_object is not None:
                    return spawned_object
            except Exception as e:
                print(f"Error: {e}")
                final_spawn_transform = self.generate_random_relative_transform(relative_entity_transform, -10, 10)
    
    def remove_object(self, self_entity: str):
        
        
        
        
        self_entity_name = self.get_name(self_entity)

        
        # Get the object
        spawned_object = self.find_actor_by_role_name(self_entity_name)
        if spawned_object is None:
            return None
        
        # Destroy the object
        spawned_object.destroy()
        return None
    
    def remove_object_close_to(self, relative_entity, reference_entity_type: EntityType, self_entity_name=None):
        
        relative_entity_name = self.get_name(relative_entity)
        
        # Get the relative entity
        relative_entity = self.find_actor_by_role_name(relative_entity_name)
        if relative_entity is None:
            return None
        
        close_objects = self.get_close_entities(relative_entity, reference_entity_type)
        
        if len(close_objects) > 0:
            object_to_remove = random.choice(close_objects)
            object_name = object_to_remove.attributes['role_name']
            self.remove_object(object_name)
            return object_to_remove
        else:
            return None
    
    def get_random_object(self, relative_entity=None, reference_entity_type: EntityType=EntityType.UNKNOWN):
        
        
        if relative_entity is not None and reference_entity_type != EntityType.UNKNOWN:
            close_objects = self.get_close_entities(relative_entity, reference_entity_type)
            if len(close_objects) > 0:
                return random.choice(close_objects)
            else:
                return None
        elif relative_entity is not None:
            return random.choice(self.get_close_entities(relative_entity, EntityType.UNKNOWN))
        elif reference_entity_type == EntityType.PEDESTRIAN:
            return random.choice(self.get_pedestrian_actors())
        elif reference_entity_type == EntityType.VEHICLE:
            return random.choice(self.get_vehicle_actors())

        return None
