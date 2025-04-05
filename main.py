"""
An AI project for the UNR ACM 2025 Hackathon

Authors: Max Clemetsen and Kevin Pettibone
Python version: 3.13.2
"""

#-----Import Section-----

from google import genai
import os

#-----Function Section-----

def main():
    """
    docsting thing goes here
    """
    print("You are using API key:", os.environ["GEMINI_KEY"])
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents="Explain AI to me in simple terms"
    )
    print(response.text)


if __name__ == "__main__":
    main()
