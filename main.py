"""
An AI project for the UNR ACM 2025 Hackathon

Authors: Max Clemetsen and Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import os
from google import genai
from gemini_gen import gemini_gen

#-----Function Section-----

def main():
    """
    docstring thing goes here
    """
    with open("llm_output.py", "w", encoding="utf-8") as f:
        f.write(gemini_gen())
    os.system("python llm_output.py")

if __name__ == "__main__":
    main()
