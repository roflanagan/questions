import numpy as np
import cmath
import matplotlib.pyplot as plt

def compute_V(a, b):
    """
    Return the complex number V = 2*w2 * cbrt(Y+U) for the tilde relation, 
    given P(A)=a, P(B)=b in (0,1).
    """
    # Signed probability gaps:
    gapA = 2*a - 1
    gapB = 2*b - 1

    # T, S, Y:
    T = (1/8)*(3 - gapA**2)*(3 - gapB**2) - 3/2
    S = -(5/32)*((9/5 - gapA**2)*(9/5 - gapB**2) - (9/5)**2 + 9)
    Y = gapA*gapB*S

    # U = sqrt(T^3 + Y^2) (potentially complex).
    val = T**3 + Y**2
    U   = complex(val, 0)**0.5

    # w2 = -1/2 - i*sqrt(3)/2
    w2 = complex(-0.5, -np.sqrt(3)/2)

    # cbrt( Y + U ) with principal root
    cval  = complex(Y,0) + U
    croot = cval**(1/3)

    # V = 2 * w2 * croot
    return 2 * w2 * croot

def figure_7_1(n_points=60):
    """
    Generate and plot Figure 7.1: the values of V in the complex plane for
    all (a,b) in [0,1]^2
    """
    a_vals = np.linspace(0,1,n_points)
    b_vals = np.linspace(0,1,n_points)
    
    re_vals = []
    im_vals = []

    for a in a_vals:
        for b in b_vals:
            val = compute_V(a,b)
            if val is not None:
                re_vals.append(val.real)
                im_vals.append(val.imag)
    
    # Plot as a scatter of points in the complex plane
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(re_vals, im_vals, s=2, color='C0', alpha=0.6)
    ax.set_xlabel(r'$\mathrm{Re}(V)$')
    ax.set_ylabel(r'$\mathrm{Im}(V)$')
    ax.set_title("$V$ in the Complex Plane, for A~B, P(A), P(B) in [0,1]")
    #ax.axhline(0,color='k',lw=0.8,alpha=0.5)
    #ax.axvline(0,color='k',lw=0.8,alpha=0.5)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    figure_7_1(n_points=601)
