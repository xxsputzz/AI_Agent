# Python Variables
Variables in Python are containers for storing data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

Variables do not need to be declared with any particular type, and can even change type after they have been set.

## Creating Variables
Python variables are created when you assign a value to them:

```python
x = 5
y = "Hello, World!"
```

## Variable Names
A variable can have a short name (like x and y) or a more descriptive name (age, carname, total_volume). Rules for Python variables:
- A variable name must start with a letter or the underscore character
- A variable name cannot start with a number
- A variable name can only contain alpha-numeric characters and underscores (A-z, 0-9, and _ )
- Variable names are case-sensitive (age, Age and AGE are three different variables)
- A variable name cannot be any of the Python keywords.

## Variable Types
In Python, variables can store data of different types, and different types can do different things.

Python has the following data types built-in by default:

- Text Type: str
- Numeric Types: int, float, complex
- Sequence Types: list, tuple, range
- Mapping Type: dict
- Set Types: set, frozenset
- Boolean Type: bool
- Binary Types: bytes, bytearray, memoryview
- None Type: NoneType

## Variable Scope
In Python, variables can have different scopes:
- Local scope: Variables defined within a function
- Global scope: Variables defined outside of functions
- Enclosing scope: For nested functions
- Built-in scope: Python's pre-defined identifiers
