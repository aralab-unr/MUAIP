import ast

# Open the file in read mode
with open('output.txt', 'r') as file:
    content = file.readlines()

# Convert the read strings back to tuples
tuple_list = [ast.literal_eval(line.strip()) for line in content]

print(tuple_list)  # Output: [(1, 2), (3, 4), (5, 6)]
