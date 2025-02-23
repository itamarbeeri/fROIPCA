import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def extract_leading_eigenpairs(X, m):
    cov_matrix = np.dot(X.T, X)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[sorted_indices[:m]]
    top_eigenvectors = eigenvectors[:, sorted_indices[:m]]

    return top_eigenvalues, top_eigenvectors


def load_dataset(seed=42):
    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data
    y = mnist.target.astype(int)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    X_shuffled, y_shuffled = shuffle(X_norm, y, random_state=seed)
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

    if len(filtered_evals) < len(estimate_vals):
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

    u_history = list()
    for i in range(N):
        Q = np.array(eigen_vecs)
        x = next(generator)
        x[np.abs(x) < 1e-12] = 0

        rho = np.dot(x, x)
        v = x / np.sqrt(rho)
        z = np.dot(Q.T, v)

        if not const_u:
            u = (B - np.sum(eigen_vals)) / (d - m)
            B += rho

        u_history.append(u)
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

    return eigen_vals, eigen_vecs, u_history


def run_fROIPCA(init_size=5000, N_updates=2000, m=10, seed=42, const_u=False, debug=False):
    x_data, y_data = load_dataset(seed)
    generator = sample_generator(x_data)
    init_dataset = np.array([next(generator) for _ in range(init_size)])

    top_eigenvalues, top_eigenvectors = extract_leading_eigenpairs(init_dataset, m)
    estimate_vals, estimate_vecs, u_history = fROIPCA(X=init_dataset, eigen_vals=top_eigenvalues,
                                                      eigen_vecs=top_eigenvectors,
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

        plt.figure()
        plt.title('u history')
        plt.plot(u_history)
        plt.show()

    return error


def menu():
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("-ne", "--n_exp", type=int, default=20, help="number of experiments")
    parser.add_argument("-s", "--init_size", type=int, default=5000, help="initial dataset size")
    parser.add_argument("-nu", "--N_updates", type=int, default=2000, help="number of update samples")
    parser.add_argument("-m", "--PC_num", type=int, default=10, help="number of principal components")
    parser.add_argument("-r", "--seed", type=int, default=42, help="random seed")
    parser.add_argument("-u", "--const_u", action="store_true", help="use constant U or dynamic U")
    parser.add_argument("-d", "--debug", action="store_true", help="debug flag")
    return parser.parse_args()

if __name__ == '__main__':
    args = menu()
    error_list = list()
    for i in range(args.n_exp):
        print(f'iteration: {i}/{args.n_exp}')
        error_list.append(run_fROIPCA(init_size=args.init_size,
                                      N_updates=args.N_updates,
                                      m=args.PC_num,
                                      seed=args.seed,
                                      const_u=args.const_u,
                                      debug=args.debug))

    print(f'error mean: {np.mean(error_list)}, error std: {np.std(error_list)}')
    print('done')
