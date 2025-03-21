import numpy as np
import matplotlib.pyplot as plt

# Tilde_relation function
def tilde_relation(P_A, P_B):
    """Simplified tilde relation from Section 7."""
    gap_A = 2 * P_A - 1
    gap_B = 2 * P_B - 1
    T = (1/8) * (3 - gap_A**2) * (3 - gap_B**2) - 3/2
    S = -(5/32) * ((9/5 - gap_A**2) * (9/5 - gap_B**2) - (9/5)**2 + 9)
    Y = gap_A * gap_B * S
    U = 1j * np.sqrt(np.abs(T**3 + Y**2))
    w2 = (-1 - 1j * np.sqrt(3)) / 2
    V = 2 * w2 * (Y + U)**(1/3)
    x_minus_ab = (np.real(V) - gap_A * gap_B) / 3
    return P_A * P_B + x_minus_ab

P_A_vals = np.linspace(0.0001, 0.9999, 100)
P_B_vals = np.linspace(0.0001, 0.9999, 100)
P_A_grid, P_B_grid = np.meshgrid(P_A_vals, P_B_vals)
P_AB_ind = P_A_grid * P_B_grid  # Independence
P_AB_tilde = tilde_relation(P_A_grid, P_B_grid)

# --- Figure 2.4: B Given A Plot ---
b_given_a = P_AB_tilde / P_A_grid
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(P_A_grid, P_B_grid, b_given_a, cmap='viridis')
ax1.set_title("P(B|A) for A~B")
ax1.set_xlabel("P(A)")
ax1.set_ylabel("P(B)")
ax1.set_zlabel("P(B|A)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(P_A_grid, P_B_grid, b_given_a, cmap='viridis')
ax2.set_title("P(B|A) for A~B")
ax2.set_xlabel("P(A)")
ax2.set_ylabel("P(B)")
ax2.set_zlabel("P(B|A)")

# Set the viewing angle
ax1.view_init(elev=15, azim=140) 
ax2.view_init(elev=15, azim=195)  # Change these to view the surface from another angle

plt.savefig("figure_2_4.png")
plt.show()
