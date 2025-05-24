"""Module for handling simple chat responses"""
from random import choice

def is_greeting(text):
    """Check if the text is ONLY a simple greeting"""
    greetings = ["hi", "hello", "hey", "hi there", "hello there", 
                 "good morning", "good afternoon", "good evening"]
    text = text.lower().strip()
    # Only treat as greeting if it's just a greeting without additional content
    is_just_greeting = any(text == greeting for greeting in greetings)
    return is_just_greeting

def get_simple_greeting():
    """Return a simple, random greeting"""
    greetings = [
        "Hi!",
        "Hello!",
        "Hi there!",
    ]
    return choice(greetings).strip()
