import numpy as np
import matplotlib.pyplot as plt

# Define the complex number
z = 2 + 3j

# Compute the cube roots
n = 3  # Cube root
roots = [z**(1/n) * np.exp(2j * np.pi * k / n) for k in range(n)]

# Extract real and imaginary parts for plotting
real_parts = [root.real for root in roots]
imag_parts = [root.imag for root in roots]

# Plot the cube roots in the complex plane
plt.figure(figsize=(6, 6))
plt.axhline(0, color='gray', linewidth=2, linestyle='-')
plt.axvline(0, color='gray', linewidth=2, linestyle='-')
plt.scatter(real_parts, imag_parts, color='blue', label='Cube Roots')
plt.scatter([z.real], [z.imag], color='red', label='2+3i')
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Cube Roots of 2 + 3i in the Complex Plane")
plt.xlim(-3.3,3.3)
plt.ylim(-3.3,3.3)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.grid(True)
plt.show()
