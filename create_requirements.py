import ast
import os

# Define the path to your Python file
python_file = 'process.py'

# List to store imported modules
imports = []

# Parse the Python file to extract import statements
with open(python_file, 'r') as file:
    tree = ast.parse(file.read(), python_file)

# Extract import statements from the parsed AST
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module)

# Remove duplicates and sort the imports
imports = sorted(set(imports))

# Write the imports to the requirements.txt file
with open('requirements.txt', 'w') as req_file:
    for imp in imports:
        req_file.write(f'{imp}\n')

print('requirements.txt file has been generated.')