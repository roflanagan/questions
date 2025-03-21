import numpy as np
import matplotlib.pyplot as plt

theta_arc = np.linspace(0, np.pi/4, 50)
arc_x = 0.15 * np.cos(theta_arc)
arc_y = 0.15 * np.sin(theta_arc)

origin = np.array([0, 0])
point1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
point2 = np.array([1/np.sqrt(2), 0])

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Plot arrows
ax.annotate("", xy=point1, xytext=origin, arrowprops=dict(arrowstyle="->", color="black", linewidth=2, mutation_scale=15))
ax.annotate("", xy=point2, xytext=origin, arrowprops=dict(arrowstyle="->", color="black", linewidth=2, mutation_scale=15))

# Labels for arrows
ax.text(0.35/np.sqrt(2), 0.42/np.sqrt(2), r"$gap(W?)=1$", fontsize=12, rotation=45, verticalalignment='bottom')
ax.text(0.35/np.sqrt(2), -0.07, r"$gap(X?)=\cos(\theta)$", fontsize=12, verticalalignment='bottom')

# Dashed x-axis
ax.axhline(0, color='black', linestyle='dashed')

# Dotted line connecting endpoints
ax.plot([point1[0]-0.01, point2[0]-0.01], [point1[1]-0.02, point2[1]], linestyle='dotted', color='black')

# Angle theta
ax.plot(arc_x, arc_y, color='black')

# Label theta
ax.text(0.06, 0.02, r"$\theta$", fontsize=12)

# Set limits and aspect ratio
ax.set_xlim(-0.1, 1)
ax.set_ylim(-0.1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Show plot
plt.show()

