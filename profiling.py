import math
import numpy as np
import time

# Generate random angles between 0 and 2π radians
angles = np.random.uniform(0, 2*np.pi, 1000000)
print(angles)

# Using math.sin
start_time = time.time()
for angle in angles:
    math.sin(angle)
end_time = time.time()
math_time = end_time - start_time

# Using np.sin
start_time = time.time()
for angle in angles:
    np.sin(angle)
end_time = time.time()
np_time = end_time - start_time

print("Time taken using math.sin:", math_time)
print("Time taken using np.sin:", np_time)
