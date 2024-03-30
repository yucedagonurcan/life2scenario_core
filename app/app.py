
from time import sleep
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light, create_theme_imgui_dark
from comm import ProgramGenerator

text_buffer = [None]*100
text_pointer = 0
prompter = ProgramGenerator()

dpg.create_context()
dpg.create_viewport(title='Chat App', width=800, height=600)

PROMPT = "i need you to remove pedestrian actor named pedestrian_close_to_standing_241 and then would you add pedestrian close to adversary and finally i need you to remove pedestrian actor named pedestrian_w_transform_429"


def block_input():
    dpg.configure_item("prompt", enabled=False)
    dpg.configure_item("send", enabled=False)
    
def unblock_input():
    dpg.configure_item("prompt", enabled=True)
    dpg.configure_item("send", enabled=True)    

def add_message(message, role):
    global text_buffer
    global text_pointer
    
    dpg.set_value(f"text_buf_{text_pointer}", f"[{role.upper():^10}]: {message}\n-----------------------------------\n")
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
    
    
    unblock_input()


with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Return, callback=send_callback)


for i in range(100):
    text_buffer[i] = ("", 0xFFFFFFFF)


with dpg.window(tag="PrimaryWindow",label="Chat", width=400, height=400):
    dpg.add_input_text(label="prompt", id="prompt", hint="Type a message")
    dpg.add_button(label="Send", id="send", callback=send_callback)
    
    # enumerate the text buffer to add each message to the chat
    for i, (message, color) in enumerate(text_buffer):
        text_buffer[i] = dpg.add_text(message, color=color, id=f"text_buf_{i}")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)

# dpg.start_dearpygui()



while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    dpg.render_dearpygui_frame()

dpg.destroy_context()