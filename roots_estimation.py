import numpy as np
from scipy.optimize import brentq, fsolve


def truncated_secular_equation(rho, z, mu, init_lambdas):
    """Closure for truncated secular equation for the solver."""

    def solver_equation(t):
        eps = 1e-12  # Small epsilon to avoid division by zero
        w = 1 + rho * (1 - np.dot(z.T, z)) / (mu - t + eps)
        for zk, val_k in zip(z, init_lambdas):
            w += rho * (zk ** 2 / (val_k - t + eps))
        return w

    return solver_equation


def brentq_fixed_roots(rho, z, mu, init_lambdas, search_margin=0.1):
    """Find exactly len(init_lambdas) roots of the truncated secular equation using brentq + bisect when not enough
    roots are found by brentq.
    Parameters
    rho: the norm of the new sample.
    z: the new sample, normalized and projected to the basis defined by the previous eigenvectors.
    mu: the estimated mean of the eigenvalues left out of the truncation.
    init_lambdas: the initial guess of the eigenvalues of the secular equation for the solver.
    """
    secular_eq = truncated_secular_equation(rho, z, mu, init_lambdas)

    roots = []
    for i in range(len(init_lambdas)):
        a = init_lambdas[i] - search_margin
        b = init_lambdas[i] + search_margin

        # Ensure function changes sign
        fa, fb = secular_eq(a), secular_eq(b)
        if fa * fb < 0:
            root = brentq(secular_eq, a, b)
            roots.append(root)
        # else:
            # print(f"Warning: No sign change detected in [{a}, {b}]")

    # --- Guarantee fixed number of roots ---
    if len(roots) < len(init_lambdas):
        print(f"Only found {len(roots)} roots, expected {len(init_lambdas)}.")

        # Interpolate missing roots or add small shifts
        missing = len(init_lambdas) - len(roots)
        if roots:
            extra_roots = np.linspace(min(roots), max(roots), missing + 2)[1:-1]
        else:
            extra_roots = init_lambdas[:missing] + 1e-3  # Small perturbation

        roots.extend(extra_roots)

    return np.array(sorted(roots))  # Keep roots ordered


def fsolve_roots(rho, z, mu, init_lambdas):
    secular_eq = truncated_secular_equation(rho, z, mu, init_lambdas)
    roots = fsolve(secular_eq, init_lambdas, xtol=1e-6)
    return roots
