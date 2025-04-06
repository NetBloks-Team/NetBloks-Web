"""
Method for generating a neural network from a prompt

Author: Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import os
from google import genai

#-----Function Section-----

def gemini_gen(ds_name: str, nn_struct: str, error_msg:str = None) -> str:
    """
    Method for generating a neural network from a prompt

    returns: str
    An entire python script that hopefully works
    """
    #print("You are using API key:", os.environ["GEMINI_KEY"])
    #Make sure to put your Gemini API key in the environment variables
    try:
        client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    except KeyError:
        return "Gemini API key not found. Please set the GEMINI_KEY environment variable."
    full_prompt = f"""
    Please generate python code for a pytorch neural network that has the following parameters:
    {nn_struct}
    The neural network class is named 'Net'
    We are training the network on the {ds_name} dataset.
    Make sure the input size is correct for the dataset. (i.e. 1 channel for B/W images like MNIST, 3 channels for RGB images like CIFAR 10)
    """
    if error_msg != None:
        full_prompt += f"\nThe following error was generated: {error_msg}\nPlease fix the code to remove this error."
        with open("llm_output.py", "r", encoding="utf-8") as f:
            full_prompt += f"\nThe previous code is:\n{f.read()}"
    full_prompt += "\nOnly give code, and do not reply with anything else. Give the code as an entire file that can be executed."
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    output = response.text.strip("```").removeprefix("python\n")
    with open("llm_output.py", "w", encoding="utf-8") as f:
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
    try:
        client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    except KeyError:
        return "Gemini API key not found. Please set the GEMINI_KEY environment variable."
    full_prompt = f"""
    Please generate feedback on the following neural network for a beginner with programming.
    The user is training the network on the {ds_name} dataset.
    Here is a json representation of the neural network:
    {nn_struct}
    Make sure to give feedback on the following:
    - The overall structure of the neural network
    - The number of neurons in each layer
    - The activation functions used.
    - Positive and negative aspects of the network.
    - Suggestions for improvement.

    Do not give any code or markdown. Only give feedback, remember it should be positive and negative.
    Do not mention training or testing the network, or errors.
    Do not mention any errors.
    Keep the feedback simple and easy to understand and less than 4 sentences.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    return response.text.replace("*", "")

def gemini_chatbot(ds_name: str, nn_struct: str, message) -> str:
    """
    Method for generating a feedback on a user's neural network

    returns: str
    A string containing feedback
    """
    #print("You are using API key:", os.environ["GEMINI_KEY"])
    #Make sure to put your Gemini API key in the environment variables
    try:
        client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    except KeyError:
        return "Gemini API key not found. Please set the GEMINI_KEY environment variable."
    full_prompt = f"""
    You previously gave feedback to a user on their neural network for the {ds_name} dataset.
    Here is the json representation of the neural network:
    {nn_struct}
    Here is a conversation with the user: {message}. Please respond and engage with the user's message.

    Do not give any code, and do not reply with anything else. Only give feedback.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    return response.text.replace("*", "")

def explain_layer(ds_name: str, nn_struct: str, layer) -> str:
    """
    Method for generating a feedback on a user's neural network

    returns: str
    A string containing feedback
    """
    try:
        client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    except KeyError:
        return "Gemini API key not found. Please set the GEMINI_KEY environment variable."
    full_prompt = f"""
    You previously gave feedback to a user on their neural network for the {ds_name} dataset.
    Here is the json representation of the neural network:
    {nn_struct}
    Please describe layer {layer} in detail. Please include the following:
    - How it work
    - What it does
    - Why it is important
    - How it relates to the rest of the network
    - How it relates to the dataset

    Do not give any code or "*"s, and do not reply with anything else. Only give feedback.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    return response.text.replace("*", "")

def explain_error(ds_name: str, nn_struct: str, error_msg:str) -> str:
    """
    Method for generating a feedback on a user's neural network

    returns: str
    A string containing feedback
    """
    try:
        client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    except KeyError:
        return "Gemini API key not found. Please set the GEMINI_KEY environment variable."
    full_prompt = f"""
    You previously gave feedback to a user on their neural network for the {ds_name} dataset.
    Here is the json representation of the neural network:
    {nn_struct}
    The following error was generated: {error_msg}
    If the error is related to the design of the neural network, explain why the error occured and how to fix it.
    If the error is not related to the design of the neural network, say "An internal error occurred. Please try again."

    Do not give any code or "*"s, and do not reply with anything else.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    return response.text.replace("*", "")

def getting_started(database, layers, activations) -> str:
    """
    Method for generating a getting started guide for the user

    returns: str
    A string containing feedback
    """
    try:
        client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    except KeyError:
        return "Gemini API key not found. Please set the GEMINI_KEY environment variable."
    full_prompt = f"""
    Please explain how to get started with the {database} dataset.
    Include the following:
    - What layers to use, and why
    - What activation functions to use, and why
    - How the layers relate to the dataset
    - Sizes of the layers. The layers and their parameters are:
    {layers}
    The activation functions are:
    {activations}

    Do not give any code or "*"s, and do not reply with anything else. Only give feedback, and keep it somewhat short.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    return response.text.replace("*", "")
