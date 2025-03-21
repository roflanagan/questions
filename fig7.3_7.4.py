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

# Calculate V and fold_gaps for each combination of a and b
for a in a_vals:
    for b in b_vals:
        V_points.append(V(a, b))



# Prepare data for the new plots
gap_a_vals = []
gap_b_vals = []
T_vals = []
Y_vals = []
S_vals = []
U_imag_vals = []
three_minus_GA_squared = []
three_minus_GB_squared = []
nine_fifths_minus_GA_squared = []
nine_fifths_minus_GB_squared = []
V_points = []
cbrt_Y_plus_U_points = []

for a in a_vals:
    for b in b_vals:
        g_a = gap(a)
        g_b = gap(b)
        G_A = g_a
        G_B = g_b

        # Calculate required values
        gap_a_vals.append(g_a)
        gap_b_vals.append(g_b)
        T_vals.append(T_(a, b))
        Y_vals.append(Y_(a, b))
        S_vals.append(S(G_A, G_B))
        U_val = U(a, b)
        U_imag_vals.append(U_val.imag)
        three_minus_GA_squared.append(3 - G_A**2)
        three_minus_GB_squared.append(3 - G_B**2)
        nine_fifths_minus_GA_squared.append(9 / 5 - G_A**2)
        nine_fifths_minus_GB_squared.append(9 / 5 - G_B**2)
        V_points.append(V(a, b))
        cbrt_Y_plus_U_points.append((Y_(a, b) + U_val)**(1/3))

# Plot graphs
plt.figure(figsize=(14, 12))

# gap(b) vs gap(a)
plt.subplot(3, 3, 1)
plt.plot(gap_a_vals, gap_b_vals, '.', alpha=0.7, markersize=3)
plt.xlabel("gap(a)")
plt.ylabel("gap(b)")
plt.title("gap(b) vs gap(a)")

# S vs gap(a)
plt.subplot(3, 3, 2)
plt.plot(gap_a_vals, S_vals, '.', alpha=0.7, markersize=2)
plt.xlabel("gap(a)")
plt.ylabel("S")
plt.title("S vs gap(a)")

# T vs gap(a)
plt.subplot(3, 3, 3)
plt.plot(gap_a_vals, T_vals, '.', alpha=0.7, markersize=2)
plt.xlabel("gap(a)")
plt.ylabel("T")
plt.title("T vs gap(a)")

# T vs S
plt.subplot(3, 3, 4)
plt.plot(S_vals, T_vals, ',', alpha=0.7)
plt.xlabel("S")
plt.ylabel("T")
plt.title("T vs S")


# S vs Y
plt.subplot(3, 3, 5)
plt.plot(Y_vals, S_vals, '.', alpha=0.7, markersize=3)
plt.xlabel("Y")
plt.ylabel("S")
plt.title("S vs Y")

# T vs Y
plt.subplot(3, 3, 6)
plt.plot(Y_vals, T_vals, '.', alpha=0.7, markersize=3)
plt.xlabel("Y")
plt.ylabel("T")
plt.title("T vs Y")

# Imag(U) vs Y
plt.subplot(3, 3, 7)
plt.plot(Y_vals, U_imag_vals, '.', alpha=0.7, markersize=3)
plt.xlabel("Y")
plt.ylabel("Imag(U)")
plt.title("Y + U in the complex plane")

# (Y + U)^(1/3) in the complex plane
plt.subplot(3, 3, 8)
plt.scatter(
    [z.real for z in cbrt_Y_plus_U_points],
    [z.imag for z in cbrt_Y_plus_U_points],
    alpha=0.7,
    s=4
)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("(Y + U)^(1/3) in the complex plane")

# V in the complex plane
plt.subplot(3, 3, 9)
plt.scatter([v.real for v in V_points], [v.imag for v in V_points], alpha=0.7, s=4)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("V in the complex plane")

plt.tight_layout()
plt.show()
# Determine the range of S and T for finer plots
S_min, S_max = min(S_vals), max(S_vals)
T_min, T_max = min(T_vals), max(T_vals)

# Define the number of slices and the resolution for the smaller plots
num_slices = 5
slice_width = (S_max - S_min) / num_slices

# Create a series of plots focusing on smaller parts of the S vs T plot
plt.figure(figsize=(12, 8))

for i in range(num_slices):
    S_start = S_min + i * slice_width
    S_end = S_start + slice_width

    # Filter the points within the current slice
    S_slice = [S for S, T in zip(S_vals, T_vals) if S_start <= S < S_end]
    T_slice = [T for S, T in zip(S_vals, T_vals) if S_start <= S < S_end]

    # Plot the current slice
    plt.subplot(2, 3, i + 1)
    plt.scatter(S_slice, T_slice, s=1, alpha=0.7)
    plt.xlabel("S")
    plt.ylabel("T")
    plt.title(f"T vs S slice {i+1}: {S_start:.2f} to {S_end:.2f}")
    plt.grid()

# Adjust layout
plt.tight_layout()
plt.show()

