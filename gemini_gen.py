"""
Method for generating a neural network from a prompt

Author: Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import os
from google import genai

#-----Function Section-----

def gemini_gen() -> str:
    """
    Method for generating a neural network from a prompt
    """
    #print("You are using API key:", os.environ["GEMINI_KEY"])
    #Make sure to put your Gemini API key in the environment variables
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    with open("nn.json", "r", encoding="utf-8") as f:
        prompt = f.read()
    full_prompt = "Please generate python code for a pytorch neural network that has the following parameters:\n" + prompt +"\nOnly give code, and do not reply with anything else. Give the code as an entire file that can be executed. The neural network class is named 'Net'"
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    output = response.text.strip("```").removeprefix("python\n")
    return output