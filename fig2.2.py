import numpy as np
import matplotlib.pyplot as plt

def tilde_relation(a, b):
    """
    Compute P(AB) for the tilde relation, given P(A)=a, P(B)=b,
    using the 'simplified' formula from Section 7.
    """
    # Handle trivial boundary cases: if a or b is 0 or 1, we can just return a*b
    # to avoid 0/0 issues in the formula. (The paper shows that at boundaries,
    # the tilde solution collapses to independence or is otherwise unconstrained.)
    if (a <= 0.0 or a >= 1.0) or (b <= 0.0 or b >= 1.0):
        return a*b

    # Signed probability gaps:
    gapA = 2*a - 1  # gap(A?)
    gapB = 2*b - 1  # gap(B?)

    # Compute T, S, Y:
    T = (1/8)*(3 - gapA**2)*(3 - gapB**2) - 3/2
    S = -(5/32)*((9/5 - gapA**2)*(9/5 - gapB**2) - (9/5)**2 + 9)
    Y = gapA*gapB*S

    # U = sqrt(T^3 + Y^2).  This can be imaginary if T^3 + Y^2 < 0.
    val = T**3 + Y**2
    # Python's complex sqrt handles sign automatically:
    U = complex(val, 0)**0.5

    # w2 = (-1 - i sqrt(3))/2
    w2 = complex(-0.5, -np.sqrt(3)/2)

    # The cube root of (Y + U), using Python's principal root:
    cval  = complex(Y, 0) + U
    croot = cval**(1/3)

    # V = 2 * w2 * croot
    V = 2 * w2 * croot

    # Real part of V:
    rV = V.real

    # Final formula: x - a*b = (rV - gapA*gapB)/3
    x = a*b + (rV - gapA*gapB)/3
    return x

# Create a mesh grid for a = P(A), b = P(B):
N = 20
a_vals = np.linspace(0, 1, N)
b_vals = np.linspace(0, 1, N)
A_mesh, B_mesh = np.meshgrid(a_vals, b_vals)

# Compute surfaces: left (independence) and right (tilde).
indep_surface = A_mesh * B_mesh

tilde_surface = np.zeros_like(indep_surface)
for i in range(N):
    for j in range(N):
        a = A_mesh[i,j]
        b = B_mesh[i,j]
        tilde_surface[i,j] = tilde_relation(a, b)

# Matplotlib 3D plotting:
fig = plt.figure(figsize=(12,5))

# Left subplot: independence
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(A_mesh, B_mesh, indep_surface, cmap='plasma', edgecolor='none')
ax1.set_title(r"$P(AB)=P(A)\,P(B)$ (Independence)")
ax1.set_xlabel("P(A)")
ax1.set_ylabel("P(B)")
ax1.set_zlabel("P(AB)")

# Right subplot: tilde
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(A_mesh, B_mesh, tilde_surface, cmap='plasma', edgecolor='none')
ax2.set_title(r"$P(AB)$ for $A\sim B$ (Tilde Relation)")
ax2.set_xlabel("P(A)")
ax2.set_ylabel("P(B)")
ax2.set_zlabel("P(AB)")

plt.tight_layout()
plt.show()
