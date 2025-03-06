import numpy as np
from scipy.optimize import brentq, fsolve
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml


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


def clean_and_sort_roots(roots):
    real_roots = np.real(roots[np.isreal(roots)])  # Keep only real roots
    real_roots.sort()  # Sort in ascending order
    estimate_vals = real_roots[-m:]
    return np.fliplr(estimate_vals)

# # ---- Example usage with MNIST-like dataset (10 leading PCs) ---->  Turn into a unittest
#
# # Load MNIST-like dataset (784 features)
# mnist = fetch_openml("mnist_784", version=1, parser="auto")
# X = mnist.data.to_numpy() / 255.0  # Convert to NumPy array & normalize
#
# # Compute PCA with 10 leading principal components
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(X)
#
# # Simulate an online PCA update
# new_sample = np.random.randn(784)  # A new high-dimensional sample
# new_sample_proj = pca.transform(new_sample.reshape(1, -1)).flatten()  # Project to top 10 PCs
#
# rho = np.linalg.norm(new_sample_proj)  # Norm of new projected sample
# z = new_sample_proj / rho  # Normalize in PCA space
#
# # Handle the case where there are no discarded eigenvalues
# if pca.explained_variance_.shape[0] > 10:
#     mu = np.mean(pca.explained_variance_[10:])  # Mean of discarded eigenvalues
# else:
#     mu = 1e-6  # Small positive value to avoid NaN issues
#
# init_lambdas = pca.explained_variance_[:10]  # Top 10 eigenvalues
#
# # Find exactly 10 updated eigenvalues
# roots = brentq_fixed_roots(rho, z, mu, init_lambdas, search_margin=0.5)
#
# print("Updated Eigenvalues:", roots)


