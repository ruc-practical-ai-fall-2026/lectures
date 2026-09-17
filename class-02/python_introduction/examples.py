# Starting Python on the Command Line

# > python --version
# > python
# > print("Hello World!")
# > exit()

"""Multiline comments

These are more comments
"""

# Printing and Hello
print('Hello World!')

# Indentation
if True:
    print("Hello")

# Arithmetic
print (2+2)

# Variables
x = 2
y = x + 3
print(y)

# Types
x = str(20)
y = int(3.3)
print(x)
print(y)

# Collections
letters = ["a", "b", "c"]
letter1, letter2, letter3 = letters

print(letters)
print(letter1, letter2, letter3)

# Functions
def my_function_f(x, y):
    """Add x and y"""
    return x + y

x = 1
y = 2
z = my_function_f(x, y)
print(z)

# Lambda
f = lambda x : x ** 2

print(f(3))

# Loops
for x in [1,2,3,]:
    print(x)

for x in 'abcd':
    print(x)

for x in range(10):
    print(x)

# Arrays
import numpy as np

x = np.array([0, 1, 2, 3, 4])

print(x[1:3]) # print from 1, up to but not including, 3
print(x[1:]) # print from 1, up to the end

# If statements
x = 5
if x > 0:
    print('X is positive')
elif x == 0:
    print('X is zero')
else:
    print('X is negative')

# Lists vs. Tuples vs. Sets vs. Dictionaries
#
# Python provides multiple types of data structures for storing data.
#
# To understand when to use each, it is important to think about if our
# data needs to be ordered and changeable.
#
# If our data needs to be ordered, this means when we store it in one
# order that order must be pre

# Lists are **ordered and changeable** and use straight brackets, []
example_list = [1,2,3]
print(example_list)
example_list[0] = 10
print(example_list)

# Tuples are **ordered and unchangeable** and use parenthesis, ()
example_tuple = (1,2,3)
print(example_tuple)
# example_tuple[0] = 10 # This would cause an error!

# Sets are **unordered and unchangeable** and do not allow duplicates, they use curly brackets, {}
vehicles = {"Car", "Truck", "Bus", "Train"}
print(vehicles) # Not the order we specified!
vehicles.add("Boat")
print(vehicles)
vehicles.add("Train") # Does not change the set - Train is already a member!
print(vehicles)

# Dictionaries are **ordered and changeable** in Python 3.7+, and do not allow duplicates
# Dictionaries take the form my_dictionary = {"key" : "value"}
song_record = {
  "artist": "Pink Floyd",
  "album": "Dark Side of the Moon",
  "title": "Breathe",
  "year": 1973
}
print(song_record)
song_record["label"] = "Capitol Records" # Adds a field to the record
print(song_record)
song_record["album"] = "Dark Side of the Moon" # Does not change the dictionary!

