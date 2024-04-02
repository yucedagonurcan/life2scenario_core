
from time import sleep, time
import dearpygui.dearpygui as dpg
from comm import ProgramGenerator
from life2scenario_model import Simulator, Life2ScenarioTaskProcessor

text_buffer = [None]*100
text_pointer = 0
prompter = ProgramGenerator()

dpg.create_context()
dpg.create_viewport(title='Life2Scenario - Chat', width=800, height=600)

PROMPT = "i need you to remove pedestrian actor named pedestrian_close_to_standing_241 and then would you add pedestrian close to adversary and finally i need you to remove pedestrian actor named pedestrian_w_transform_429"
TASK_LIST=[ "- kRANDOMPED_1 = GET_RANDOM_PEDESTRIAN()",
            "- kVEH_1 = GET_RANDOM_VEHICLE(related_entity=kRANDOMPED_1)",
            "- REMOVE_VEHICLE(self_entity=kVEH_1)",
            "- ADD_PEDESTRIAN(related_entity='adversary')",
            "- kVEH_2 = GET_RANDOM_VEHICLE()",
            "- ADD_PEDESTRIAN(related_entity=kVEH_2)"
            ]

task_processor = Life2ScenarioTaskProcessor()
# task_processor.process_task_list(tasklist=TASK_LIST)

SPACER="="*40
SUBSPACER="-"*20

def block_input():
    dpg.configure_item("prompt", enabled=False)
    dpg.configure_item("send", enabled=False)
    
def unblock_input():
    dpg.configure_item("prompt", enabled=True)
    dpg.configure_item("send", enabled=True)    

def add_message(message, role):
    global text_buffer
    global text_pointer
    
    dpg.set_value(f"text_buf_{text_pointer}", f"[{role.upper():^10}]: {message}\n{SPACER}\n")
    text_pointer = (text_pointer + 1) % 100
    
        
def send_callback(sender, app_data):
    
    if dpg.get_value("prompt") == "":
        return
    
    block_input()
    
    # Get the message from the input box
    message = dpg.get_value("prompt")
    
    # Append the message to the chat
    add_message(message, "user")
    
    # Clear the input box
    dpg.set_value("prompt", "")
    
    unit_tasks = prompter.generate(message)
    add_message(f"I'm thinking about it, wait a sec...", "system")
    
    
    
    system_response = "Here are the tasks I have identified: \n"
    for idx, task in enumerate(unit_tasks):
        system_response += f"\t{idx+1}. {task}\n"
    add_message(system_response, "system")
    
    for idx, task in enumerate(unit_tasks):
        task_processor.process_task(task)
        sleep(1)
        add_message(f"Task {idx+1} completed successfully", "system")
    
    unblock_input()


with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Return, callback=send_callback)


for i in range(100):
    text_buffer[i] = ("", 0xFFFFFFFF)


with dpg.window(tag="PrimaryWindow",label="Chat", width=400, height=400):
    dpg.add_text(label="simulator_info", id="simulator")
    dpg.add_spacing(count=5)
    
    dpg.add_input_text(label="prompt", id="prompt", hint="Type a message")
    dpg.add_button(label="Send", id="send", callback=send_callback)
    
    # enumerate the text buffer to add each message to the chat
    for i, (message, color) in enumerate(text_buffer):
        text_buffer[i] = dpg.add_text(message, color=color, id=f"text_buf_{i}")
        

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)

# dpg.start_dearpygui()

simulator = Simulator()

prev_check_time = time()

while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    
    cur_time = time()
    if cur_time - prev_check_time > 2:
        prev_check_time = cur_time
        if simulator.connection_established:
            vehicle_actors = simulator.get_vehicle_actors()
            pedestrian_actors = simulator.get_pedestrian_actors()
            
            vehicle_actors = [(actor.attributes["role_name"], actor.get_transform()) for actor in vehicle_actors]
            pedestrian_actors = [(actor.attributes["role_name"], actor.get_transform()) for actor in pedestrian_actors]
            
            
            simulator_info = f"{SPACER}\n{'SIMULATOR INFO':^40}\n{SPACER}"
            simulator_info += f"\n{SUBSPACER}\nVehicle Actors: [{len(vehicle_actors)}]\n{SUBSPACER}"
            for idx, actor in enumerate(vehicle_actors):
                simulator_info += f"\n\t{idx+1}. {actor[0]:<40} {actor[1]}"
            
            simulator_info += f"\n{SUBSPACER}\nPedestrian Actors: [{len(pedestrian_actors)}]\n{SUBSPACER}"
            for idx, actor in enumerate(pedestrian_actors):
                simulator_info += f"\n\t{idx+1}. {actor[0]:<40} {actor[1]}"
            
            dpg.set_value("simulator", f"{simulator_info}")
            
        else:
            print("No connection to simulator")
            simulator.connect_simulator()
    
    dpg.render_dearpygui_frame()

dpg.destroy_context()