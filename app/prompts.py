INCONTEXT_EXS=[
"""
def ADD_PEDESTRIAN(self, location=None, related_entity=None, self_entity_name=None):
def GET_RANDOM_PEDESTRIAN(self, related_entity=None):
def GET_RANDOM_VEHICLE(self, related_entity=None):
def REMOVE_PEDESTRIAN(self, related_entity=None, self_entity=None):
def REMOVE_VEHICLE(self, related_entity=None, self_entity=None):
""",
"""
INPUT: i would like to lead you to add pedestrian close to adversary and then could you add pedestrian close to hero and then please add pedestrian close to adversary
OUTPUT: 
- ADD_PEDESTRIAN(related_entity='adversary')
- ADD_PEDESTRIAN(related_entity='hero')
- ADD_PEDESTRIAN(related_entity='adversary')
""",
"""
INPUT: i would like to tell you to remove pedestrian actor named pedestrian_close_to_adversary_1 then i would like you to demonstrate add pedestrian close to hero and then i would like to lead you to add pedestrian at location (-62.51, 82.89, 0.5)
OUTPUT: 
- REMOVE_PEDESTRIAN(self_entity='pedestrian_close_to_adversary_1')
- ADD_PEDESTRIAN(related_entity='hero')
- ADD_PEDESTRIAN(location=(-62.51, 82.89, 0.5))
""",
"""
INPUT: i would like to lead you to add pedestrian close to standing then i would like to ask you to add pedestrian close to standing and finally please add pedestrian close to hero
OUTPUT:
- ADD_PEDESTRIAN(related_entity='standing')
- ADD_PEDESTRIAN(related_entity='standing')
- ADD_PEDESTRIAN(related_entity='hero')
""",
"""
INPUT: i would like to direct you to add pedestrian at location (90.54, -71.08, 0.5) and i want you to remove pedestrian close to standing and finally could you remove pedestrian actor named pedestrian_close_to_standing_175
OUTPUT:
- ADD_PEDESTRIAN(location=(90.54, -71.08, 0.5))
- REMOVE_PEDESTRIAN(self_entity='pedestrian_close_to_standing')
- REMOVE_PEDESTRIAN(self_entity='pedestrian_close_to_standing_175')
""",
"""
INPUT: i would like to instruct you to add pedestrian named pedestrian_close_to_standing_241 close to adversary
OUTPUT:
- ADD_PEDESTRIAN(related_entity='hero', self_entity='pedestrian_close_to_standing_241')
""",
"""
INPUT: i would like to instruct you to remove random pedestrian close to hero
OUTPUT:
- kPED_1 = GET_RANDOM_PEDESTRIAN(related_entity='hero')
- REMOVE_PEDESTRIAN(self_entity=kPED_1)
""",
"""
INPUT: i would like to instruct you to add a pedestrian close to a vehicle object
OUTPUT:
- kVEH_1 = GET_RANDOM_VEHICLE()
- ADD_PEDESTRIAN(related_entity=kVEH_1)
""",
"""
INPUT: i would like to instruct you to remove a random vehicle object close to a random pedestrian object
OUTPUT:
- kPED_1 = GET_RANDOM_PEDESTRIAN()
- kVEH_1 = GET_RANDOM_VEHICLE(related_entity=kPED_1)
- REMOVE_VEHICLE(self_entity=kVEH_1)
""",
"""
INPUT: i would like to instruct you to remove a random vehicle object close to a random pedestrian object and i would like you to add a pedestrian close to the adversary vehicle and then please add a pedestrian close to the random vehicle object
OUTPUT:
- kRANDOMPED_1 = GET_RANDOM_PEDESTRIAN()
- kVEH_1 = GET_RANDOM_VEHICLE(related_entity=kRANDOMPED_1)
- REMOVE_VEHICLE(self_entity=kVEH_1)
- ADD_PEDESTRIAN(related_entity='adversary')
- kVEH_2 = GET_RANDOM_VEHICLE()
- ADD_PEDESTRIAN(related_entity=kVEH_2)
"""
]