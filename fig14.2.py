import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# 1) Helper functions: state creation, normalization, Bloch-sphere coordinates
##############################################################################

def normalize(psi):
    """Normalize a 2D complex state vector."""
    norm = np.linalg.norm(psi)
    if norm < 1e-14:
        raise ValueError("Cannot normalize near-zero vector.")
    return psi / norm

def bloch_coords(psi):
    """
    Convert a normalized 2D complex state vector |psi> = (a, b) to Bloch-sphere
    coordinates (x,y,z).  We use the standard formulas:
       x = 2 Re(a^* b)
       y = 2 Im(a^* b)
       z = |a|^2 - |b|^2.
    """
    a, b = psi
    x = 2.0 * np.real(np.conjugate(a)*b)
    y = 2.0 * np.imag(np.conjugate(a)*b)
    z = np.abs(a)**2 - np.abs(b)**2
    return np.array([x, y, z], dtype=float)

def inner_product(psi1, psi2):
    """Dirac-style inner product <psi1|psi2> = conj(psi1)^T . psi2."""
    return np.vdot(psi1, psi2)

def bloch_angle(psi1, psi2):
    """
    The angular distance on the Bloch sphere between two normalized states:
        angle = 2 * arccos(|<psi1|psi2>|).
    """
    ov = inner_product(psi1, psi2)
    mag = np.abs(ov)
    if mag > 1.0:  # numerical safeguard
        mag = 1.0
    return 2.0 * np.arccos(mag)

##############################################################################
# 2) Define two states |psi1>, |psi2> in Bloch form and convert to 2D vectors
##############################################################################

def state_on_bloch_sphere(theta, phi):
    """
    Returns the 2D complex state vector:
        |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    """
    return np.array([
        np.cos(theta/2),
        np.exp(1j*phi)*np.sin(theta/2)
    ], dtype=complex)

# Choose two "nice" states that are not too close and not diametrically opposite
theta1, phi1 = 0.7, 1.2
theta2, phi2 = 2.0, -0.5

psi1 = normalize(state_on_bloch_sphere(theta1, phi1))
psi2 = normalize(state_on_bloch_sphere(theta2, phi2))

# For clarity in the plot, let's confirm they're neither identical nor opposite
angle_12 = bloch_angle(psi1, psi2)
print(f"Angle between psi1 and psi2 on Bloch sphere = {angle_12:.3f} radians.")

##############################################################################
# 3) Compute some key points:
#    - naive sum:   psi_sum = (psi1 + psi2) / ||...||
#    - short-arc midpoint:  phi = - arg(<psi1|psi2>)
##############################################################################

# Naive sum:
psi_sum_unnorm = psi1 + psi2
psi_sum = normalize(psi_sum_unnorm)

# Short-arc midpoint:  |psi_mid> ~ |psi1> + e^{-i arg(<psi1|psi2>)} |psi2>
overlap = inner_product(psi1, psi2)   # <psi1|psi2>
phi_mid = -np.angle(overlap)          # - arg(<psi1|psi2>)
psi_mid_unnorm = psi1 + np.exp(1j*phi_mid)*psi2
psi_mid = normalize(psi_mid_unnorm)

##############################################################################
# 4) Plot the entire great circle: psi(phi) = |psi1> + e^{i phi} |psi2>
##############################################################################

n_points = 200
phis = np.linspace(0, 2*np.pi, n_points, endpoint=False)

circle_xyz = []
for ph in phis:
    unnorm = psi1 + np.exp(1j*ph)*psi2
    circle_xyz.append(bloch_coords(normalize(unnorm)))
circle_xyz = np.array(circle_xyz).T  # shape = (3, n_points)

##############################################################################
# 5) Plot the *short* geodesic arc* by "slerp":
#    The standard formula for the geodesic from |psi1> to e^{i phi_opt}|psi2> is:
#        |psi(t)> = [sin((1-t)*alpha)/sin(alpha)] * |psi1>
#                   + e^{i phi_opt} [sin(t*alpha)/sin(alpha)] * |psi2>
#    for t in [0, 1], where alpha = angle(|psi1>,|psi2>) on Bloch sphere.
##############################################################################

alpha = angle_12

# The relative phase that places them on the short arc is phi_opt = - arg(<psi1|psi2>)
phi_opt = -np.angle(inner_product(psi1, psi2))

def slerp(psi1, psi2, alpha, phi_opt, t):
    """
    Return a point on the geodesic from psi1 to e^{i phi_opt} psi2
    using spherical linear interpolation parameter t in [0,1].
    """
    # Leading coefficients
    c1 = np.sin((1.0 - t)*alpha) / np.sin(alpha)
    c2 = np.exp(1j*phi_opt) * (np.sin(t*alpha) / np.sin(alpha))
    return normalize(c1 * psi1 + c2 * psi2)

n_arc = 50
tvals = np.linspace(0, 1, n_arc)
arc_xyz = []
for t in tvals:
    psi_arc = slerp(psi1, psi2, alpha, phi_opt, t)
    arc_xyz.append(bloch_coords(psi_arc))
arc_xyz = np.array(arc_xyz).T  # shape = (3, n_arc)

##############################################################################
# 6) Create a 3D plot: Bloch sphere + circle + short arc + points
##############################################################################

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# -- Draw a semi-transparent sphere as a backdrop --
u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2*np.pi, 30)
xs = np.outer(np.sin(u), np.cos(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.cos(u), np.ones_like(v))
ax.plot_surface(xs, ys, zs, color='lightgray', alpha=0.1, edgecolor='none')

# -- Plot the entire great circle in black --
ax.plot(circle_xyz[0], circle_xyz[1], circle_xyz[2], 'k-', lw=1, label="Full circle: $|\psi_1> + e^{i\phi}|\psi_2>$")

# -- Plot the short arc (red) from psi1 to e^{i phi_opt} psi2 --
ax.plot(arc_xyz[0], arc_xyz[1], arc_xyz[2], 'r-', lw=3, label="Short geodesic arc")

# -- Add points: psi1, psi2, psi_sum, psi_mid --
p1 = bloch_coords(psi1)
p2 = bloch_coords(psi2)
psum = bloch_coords(psi_sum)
pmid = bloch_coords(psi_mid)

ax.scatter(*p1, color='b', s=80, label="$|\psi_1>$")
ax.scatter(*p2, color='g', s=80, label="$|\psi_2>$")
ax.scatter(*psum, color='m', marker='^', s=80, label="$|\psi_1> + |\psi_2>$")
ax.scatter(*pmid, color='orange', marker='o', s=100, label="$|\psi_1> + e^{-arg(<\psi_1|\psi_2>)}|\psi_2>$")

# Cosmetic adjustments
ax.set_xlim([-1.05, 1.05])
ax.set_ylim([-1.05, 1.05])
ax.set_zlim([-1.05, 1.05])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Bloch sphere: Great circle & short-arc midpoint")

ax.legend()
plt.tight_layout()
plt.show()
