# Python Functions
Python functions are blocks of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing.

## Defining a Function
In Python, a function is defined using the `def` keyword followed by the function name and parentheses (()).

```python
def my_function():
    print("Hello from a function")
```

## Calling a Function
To call a function, use the function name followed by parentheses:

```python
my_function()  # This calls the function
```

## Parameters and Arguments
A parameter is the variable listed inside the parentheses in the function definition.
An argument is the value that is sent to the function when it is called.

```python
def my_function(fname):  # fname is a parameter
    print(f"Hello {fname}")

my_function("John")  # "John" is an argument
```

## Default Parameter Value
If we call the function without argument, it uses the default value:

```python
def my_function(country = "Norway"):
    print(f"I am from {country}")

my_function("Sweden")  # I am from Sweden
my_function()  # I am from Norway
```

## Return Values
To let a function return a value, use the return statement:

```python
def multiply(x):
    return 5 * x

print(multiply(3))  # 15
print(multiply(5))  # 25
```

## Lambda Functions
A lambda function is a small anonymous function that can take any number of arguments, but can only have one expression.

```python
x = lambda a : a + 10
print(x(5))  # 15

# Lambda functions with multiple arguments
x = lambda a, b : a * b
print(x(5, 6))  # 30
```

## Recursive Functions
Python also accepts function recursion, which means a defined function can call itself.

```python
def factorial(x):
    if x == 1:
        return 1
    else:
        return x * factorial(x-1)

print(factorial(5))  # 120
```
