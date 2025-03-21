import numpy as np
import cmath
import matplotlib.pyplot as plt

def Y(a, b):
    return -20 * a**3 * b**3 + 30 * a**3 * b**2 - 6 * a**3 * b - 2 * a**3 + 30 * a**2 * b**3 - 45 * a**2 * b**2 + 9 * a**2 * b + 3 * a**2 - 6 * a * b**3 + 9 * a * b**2 - 9 * a * b + 3 * a - 2 * b**3 + 3 * b**2 + 3 * b - 2

def T(a, b):
    return 2 * a**2 * b**2 - 2 * a**2 * b - a**2 - 2 * a * b**2 + 2 * a * b + a - b**2 + b - 1

def Y_from_gaps(G_A, G_B):
    return G_A * G_B * S(G_A, G_B)

def S(G_A, G_B):
    return -(5/32) * ((9/5 - G_A**2) * (9/5 - G_B**2) - (9/5)**2 + 9)

def T_from_gaps(G_A, G_B):
    return (1/8) * (3 - G_A**2) * (3 - G_B**2) - 3/2

def gap(a):
    return (2*a - 1)

def T_(a, b):
    return T_from_gaps(gap(a), gap(b))

def Y_(a, b):
    return Y_from_gaps(gap(a), gap(b))

def U(a, b):
    return cmath.sqrt(T_(a, b)**3 + Y_(a, b)**2)

def V(a, b):
    w1 = -(1 + cmath.sqrt(3) * 1j) / 2
    return w1 * 2 * (Y_(a, b) + U(a, b))**(1/3)

def fold_gaps(a, b):
    g_a = 2 * a - 1
    g_b = 2 * b - 1
    if g_a < g_b:
        g_a, g_b = g_b, g_a
    if g_b > -g_a:
        g_a, g_b = -g_b, -g_a
    return g_a, g_b

# Generate data
a_vals = np.linspace(0, 1, 5000)
b_vals = np.arange(0, 1.1, 0.1)

# Prepare to collect points
V_points = []
fold_points = []

# Calculate V and fold_gaps for each combination of a and b
for a in a_vals:
    for b in b_vals:
        V_points.append(V(a, b))
        g_a, g_b = fold_gaps(a, b)
        fold_points.append(g_a + 1j * g_b)

# Plot results
plt.figure(figsize=(10, 10))

plt.grid()
# Plot fold_gaps points
plt.scatter([f.real for f in fold_points], [f.imag for f in fold_points], label="gap(a) + i gap(b), folded", alpha=0.7, color='red')

# Plot V points
plt.scatter([v.real for v in V_points], [v.imag for v in V_points], label="V(a, b)", alpha=0.7)

# Labels and legend
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Complex Plane: gap(a) + i gap(b) Folded Along Main Diagonals and V(a, b)")
plt.legend()
plt.show()
