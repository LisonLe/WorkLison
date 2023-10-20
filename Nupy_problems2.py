# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:39:15 2023

@author: lison
"""
import numpy as np

def exo1():
vector=np.array(list(range(21)))

for i in range(9, 16):
    vector[i] *= -1
    
print(vector)

import numpy as np

def exo2():
vector=np.array(range(15,56))
    
for i in range(1,len(vector)-1):
    print(vector[i])
    
import numpy as np

def exo3():
x=np.empty((3,4))
print(x)

def exo4():
import numpy as np
vector=np.linspace(5,50,10)
print(vector)

def exo5():
vector=np.linspace(0,10,5)
print(vector)

def exo6():
import numpy as np

vector1 = np.array([1, 2, 3, 4])
vector2 = np.array([5, 6, 7, 8])

result = vector1 * vector2

print(result)

def exo7():
import numpy as np

matrix = np.empty((3, 4))

value = 10
for i in range(3):
    for j in range(4):
        matrix[i, j] = value
        value += 1

print(matrix)

def exo8():
import numpy as np

matrix = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

num_rows = len(matrix)
num_columns = len(matrix[0])  

print(num_rows)
print(num_columns)

def exo9():
import numpy as np

matrix = np.zeros((4, 4))

for i in range(4):
    matrix[i, i] = 0
    matrix[i, (i + 1) % 4] = 1

print(matrix)

def exo10():
import numpy as np
array=np.array(list(range(5)))
array2=np.array(list(range(5)))
array3=np.empty((5))

for i in range(len(array)):
    if array[i]==array2[i]:
        array3[i]=array[i]
                        
print(array3)

def exo11():
import numpy as np

vector1 = np.array([10, 10, 20, 20, 30, 30])
vector2 = np.array([[1, 1], [2, 3]])

unique1 = np.unique(vector1)
unique2 = np.unique(vector2)

print(unique1)
print(unique2)

def exo12():
import numpy as np

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

cross = np.cross(vector1, vector2)

print(cross)