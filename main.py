import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
thesoly

def extract_leading_eigenpairs(X, m):
    cov_matrix = np.dot(X.T, X)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[sorted_indices[:m]]
    top_eigenvectors = eigenvectors[:, sorted_indices[:m]]

    return top_eigenvalues, top_eigenvectors


def load_dataset():
    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data
    y = mnist.target.astype(int)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    X_shuffled, y_shuffled = shuffle(X_norm, y, random_state=42)

    print("Dataset loaded, standardized, and shuffled.")
    return X_shuffled, y_shuffled


def sample_generator(X):
    for i in range(X.shape[0]):
        yield X[i, :]


def get_init_dataset(x_data, n=1000):
    return x_data[:n, :]


def filter_components(estimate_vals, estimate_vecs, inital_vals):
    max_eigenvalue = 2 * np.max(inital_vals)
    min_eigenvalue = 0.5 * np.min(inital_vals)

    mask = (estimate_vals >= min_eigenvalue) & (estimate_vals <= max_eigenvalue)  # Keep only valid eigenvalues
    filtered_evals = estimate_vals[mask]
    filtered_evecs = estimate_vecs[:, mask]  # Keep corresponding eigenvectors

    print(f'detected {len(filtered_evals)}/{len(estimate_vals)} legal eigenvalues')
    return filtered_evals, filtered_evecs


def frobenius_error(true_eigenpairs, approx_eigenpairs, m):
    true_vals, true_vecs = true_eigenpairs
    approx_vals, approx_vecs = approx_eigenpairs

    A_m = sum(true_vals[i] * np.outer(true_vecs[:, i], true_vecs[:, i]) for i in range(m))
    A_m_approx = sum(approx_vals[i] * np.outer(approx_vecs[:, i], approx_vecs[:, i]) for i in range(m))

    return (np.linalg.norm(A_m - A_m_approx, 'fro') ** 2) / (np.linalg.norm(A_m, 'fro') ** 2)


def fROIPCA(X, eigen_vals, eigen_vecs, N, generator, u=0, const_u=False):
    def truncated_secular_equation(t):
        eps = 0.01
        w = 1 + rho * (1 - np.dot(z.T, z)) / (u - t + eps)
        for zk, val_k in zip(z, eigen_vals):
            w += rho * (zk ** 2 / (val_k - t + eps))
        return w

    d = X.shape[1]
    m = len(eigen_vals)
    B = np.trace(np.dot(X.T, X))

    for i in range(N):
        Q = np.array(eigen_vecs)
        x = next(generator)
        x[np.abs(x) < 1e-12] = 0

        rho = np.dot(x, x)
        v = x / np.sqrt(rho)
        z = np.dot(Q.T, x)

        if not const_u:
            u = (B - np.sum(eigen_vals)) / (d - m)
            B += rho

        roots = fsolve(truncated_secular_equation, eigen_vals, xtol=1e-6)

        real_roots = np.real(roots[np.isreal(roots)])  # Keep only real roots
        real_roots.sort()  # Sort in ascending order
        estimate_vals = real_roots[-m:]

        estimate_vecs = list()
        term1 = np.sum([np.dot(vec.T, v) * vec for vec in eigen_vecs.T])
        for vec, val, est_val in zip(eigen_vecs.T, eigen_vals, estimate_vals):
            term2 = (1 / np.dot(vec.T, v) ** 2) * (val - est_val) / (u - est_val)
            term3 = np.dot(vec.T, v) * v - np.dot(vec.T, v) * term1
            estimate_vecs.append(vec + term2 * term3)

        estimate_vecs = np.array([vec / np.linalg.norm(vec) for vec in estimate_vecs]).T

        eigen_vals = estimate_vals
        eigen_vecs = estimate_vecs

    return eigen_vals, eigen_vecs


def run_fROIPCA(init_size=5000, N_updates=2000, m=10, const_u=False, debug=False):
    x_data, y_data = load_dataset()
    generator = sample_generator(x_data)
    init_dataset = np.array([next(generator) for _ in range(init_size)])

    top_eigenvalues, top_eigenvectors = extract_leading_eigenpairs(init_dataset, m)
    estimate_vals, estimate_vecs = fROIPCA(X=init_dataset, eigen_vals=top_eigenvalues, eigen_vecs=top_eigenvectors,
                                           N=N_updates,
                                           generator=generator, u=0,
                                           const_u=const_u)

    true_eigenvalues, true_eigenvectors = extract_leading_eigenpairs(x_data[:init_size + N_updates], m)
    filtered_evals, filtered_evecs = filter_components(estimate_vals, estimate_vecs, true_eigenvalues)
    error = frobenius_error((true_eigenvalues, true_eigenvectors), (filtered_evals, filtered_evecs),
                            m=len(filtered_evals))

    if debug:
        true_eigenvalues.sort()

        plt.figure()
        plt.title(f'Eigenvalues - error {error}')
        plt.plot(true_eigenvalues[:len(filtered_evals)], label='True Eigenvalues')
        plt.plot(filtered_evals, label='Filtered Eigenvalues')
        plt.legend()  # Add legend to the plot
        plt.show()

    return error


def run_multiple_experiments(n_exp=20):
    error_u0 = list()
    for i in range(n_exp):
        error_u0.append(run_fROIPCA(const_u=True))

    error_u_mean = list()
    for i in range(n_exp):
        error_u_mean.append(run_fROIPCA(const_u=False))

    print(f'mean error for u=0 is: {np.mean(error_u0)}')
    print(f'mean error for u_mean is: {np.mean(error_u_mean)}')


if __name__ == '__main__':
    run_multiple_experiments()
