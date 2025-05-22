"""Module for handling simple chat responses"""
from random import choice

def is_greeting(text):
    """Check if the text is a simple greeting"""
    greetings = ["hi", "hello", "hey", "hi there", "hello there", 
                 "good morning", "good afternoon", "good evening"]
    text = text.lower().strip()
    return any(text.startswith(greeting) for greeting in greetings)

def get_simple_greeting():
    """Return a simple, random greeting"""
    greetings = [
        "Hi!",
        "Hello!",
        "Hi there!",
    ]
    return choice(greetings).strip()
