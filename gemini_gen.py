"""
Method for generating a neural network from a prompt

Author: Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import os
from google import genai

#-----Function Section-----

def gemini_gen(ds_name: str, nn_struct: str) -> str:
    """
    Method for generating a neural network from a prompt

    returns: str
    An entire python script that hopefully works
    """
    #print("You are using API key:", os.environ["GEMINI_KEY"])
    #Make sure to put your Gemini API key in the environment variables
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    full_prompt = ""
    full_prompt += "Please generate python code for a pytorch neural network that has the following parameters:\n"
    full_prompt += nn_struct
    full_prompt += "\nOnly give code, and do not reply with anything else. Give the code as an entire file that can be executed."
    full_prompt += "The neural network class is named 'Net'"
    full_prompt += f"We are training the network on the {ds_name} dataset."
    full_prompt += "Make sure the input size is correct for the dataset. (i.e. 1 channel for B/W images like MNIST, 3 channels for RGB images like CIFAR 10)"
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    output = response.text.strip("```").removeprefix("python\n")
    with open("llm_output.py", "w") as f:
            f.write(output)
    return output

def gemini_fb(ds_name: str, nn_struct: str) -> str:
    """
    Method for generating a feedback on a user's neural network

    returns: str
    A string containing feedback
    """
    #print("You are using API key:", os.environ["GEMINI_KEY"])
    #Make sure to put your Gemini API key in the environment variables
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    full_prompt = ""
    full_prompt += "Please generate python code for a pytorch neural network that has the following parameters:\n"
    full_prompt += nn_struct
    full_prompt += f"\nWe are training the network on the {ds_name} dataset"
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    return response.text