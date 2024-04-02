import openai
import os
import numpy as np

from prompts import INCONTEXT_EXS

class ProgramGenerator():
    def __init__(self,temperature=0.7,top_p=0.5,prob_agg='mean'):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        
    def create_prompt(self,input):
        prompt = "\n".join(INCONTEXT_EXS) + "\n"
        prompt += f"INPUT: {input}\nOUTPUT:"
        return prompt
        
        
    def generate(self,input):
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "user", "content": self.create_prompt(input)}
            ]
        )

        prog = response.choices[0].message.content.split("\n")
        return prog