import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from roots_estimation import fsolve_roots, brentq_fixed_roots


def extract_leading_eigenpairs(X, m):
    cov_matrix = np.dot(X.T, X)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[sorted_indices[:m]]
    top_eigenvectors = eigenvectors[:, sorted_indices[:m]]

    return top_eigenvalues, top_eigenvectors


def load_dataset(dataset_name='mnist_784', seed=42):
    # Load the MNIST dataset
    mnist = fetch_openml(dataset_name, version=1, as_frame=False, parser="liac-arff")
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


def fROIPCA(X, eigen_vals, eigen_vecs, N, generator, mu=0, const_mu=False, qr_every=0, optimizer='fsolve'):

    d = X.shape[1]
    m = len(eigen_vals)
    B = np.trace(np.dot(X.T, X))

    u_history = list()
    update_times = list()
    for i in range(N):
        t0 = time()

        Q = np.array(eigen_vecs)
        x = next(generator)
        x[np.abs(x) < 1e-12] = 0

        rho = np.dot(x, x)
        v = x / np.sqrt(rho)
        z = np.dot(Q.T, v)

        if not const_mu:
            mu = (B - np.sum(eigen_vals)) / (d - m)
            B += rho

        u_history.append(mu)
        eigen_vals_init_guess = eigen_vals + np.random.uniform(-0.01, 0.01, size=m)            # Initial guess for the solver
        if optimizer == 'fsolve':
            roots = fsolve_roots(rho, z, mu, eigen_vals_init_guess)
        else:
            roots = brentq_fixed_roots(rho, z, mu, eigen_vals_init_guess, search_margin=0.01)

        real_roots = np.real(roots[np.isreal(roots)])  # Keep only real roots
        real_roots.sort()  # Sort in ascending order
        estimate_vals = real_roots[-m:]

        estimate_vecs = list()
        term1 = np.sum([np.dot(vec.T, v) * vec for vec in eigen_vecs.T])
        for vec, val, est_val in zip(eigen_vecs.T, eigen_vals, estimate_vals):
            term2 = (1 / np.dot(vec.T, v) ** 2) * (val - est_val) / (mu - est_val)
            term3 = np.dot(vec.T, v) * v - np.dot(vec.T, v) * term1
            estimate_vecs.append(vec + term2 * term3)

        estimate_vecs = np.array([vec / np.linalg.norm(vec) for vec in estimate_vecs]).T

        eigen_vals = estimate_vals
        eigen_vecs = estimate_vecs


        # added orthonormalization and normalization on the real vectors according to the MATLAB implementation
        if qr_every > 0:
            if i % qr_every == 0:
                orth_vecs, _ = np.linalg.qr(eigen_vecs)
                eigen_vecs = orth_vecs
            else:
                norm_vecs = np.real(eigen_vecs) / np.linalg.norm(eigen_vecs, axis=0)[np.newaxis, :]
                eigen_vecs = norm_vecs

        update_times.append(time() - t0)
    return eigen_vals, eigen_vecs, u_history, update_times


def run_fROIPCA(init_size=5000, N_updates=2000, m=10, seed=42, const_u=False, qr_every=0, optimizer='fsolve', dataset_name='mnist_784', debug=True):
    x_data, y_data = load_dataset(dataset_name, seed)
    generator = sample_generator(x_data)
    init_dataset = np.array([next(generator) for _ in range(init_size)])

    top_eigenvalues, top_eigenvectors = extract_leading_eigenpairs(init_dataset, m)
    estimate_vals, estimate_vecs, u_history, update_times = fROIPCA(X=init_dataset, eigen_vals=top_eigenvalues,
                                                                    eigen_vecs=top_eigenvectors,
                                                                    N=N_updates,
                                                                    generator=generator,
                                                                    mu=0,
                                                                    const_mu=const_u,
                                                                    qr_every=qr_every,
                                                                    optimizer=optimizer)

    mean_update_time = np.mean(update_times)
    true_eigenvalues, true_eigenvectors = extract_leading_eigenpairs(x_data[:init_size + N_updates], m)
    error = frobenius_error((true_eigenvalues, true_eigenvectors), (estimate_vals, estimate_vecs),
                            m=len(estimate_vals))

    if debug:
        true_eigenvalues.sort()

        plt.figure()
        plt.title(f'Eigenvalues - error {error}')
        plt.plot(true_eigenvalues[:len(estimate_vals)], label='True Eigenvalues')
        plt.plot(estimate_vals, label='Filtered Eigenvalues')
        plt.legend()  # Add legend to the plot
        plt.show()

        plt.figure()
        plt.title('u history')
        plt.plot(u_history)
        plt.show()

    return error, mean_update_time


def menu():
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("-ne", "--n_exp", type=int, default=20, help="number of experiments")
    parser.add_argument("-s", "--init_size", type=int, default=5000, help="initial dataset size")
    parser.add_argument("-nu", "--N_updates", type=int, default=2000
                        , help="number of update samples")
    parser.add_argument("-m", "--PC_num", type=int, default=10, help="number of principal components")
    parser.add_argument("-r", "--seed", type=int, default=42, help="random seed")
    parser.add_argument("-u", "--const_u", action="store_true", help="use constant U or dynamic U")
    parser.add_argument("-d", "--debug", action="store_true", help="debug flag")
    parser.add_argument("-qr", "--qr_every", type=int, default=0, help="Perform QR orthogonalisation every")
    parser.add_argument("-opt", "--optimizer", type=str, default="fsolve", help="optimizer type")
    parser.add_argument("-ds", "--dataset_name", type=str, default="mnist_784", help="mnist_784, superconduct, poker-hand")
    return parser.parse_args()


if __name__ == '__main__':
    args = menu()
    error_list, mean_update_times = list(), list()
    for i in range(args.n_exp):
        print(f'iteration: {i}/{args.n_exp}')
        error, mean_update_time = run_fROIPCA(init_size=args.init_size,
                                              N_updates=args.N_updates,
                                              m=args.PC_num,
                                              seed=i,
                                              const_u=args.const_u,
                                              qr_every=args.qr_every,
                                              optimizer=args.optimizer,
                                              dataset_name=args.dataset_name,
                                              debug=args.debug
                                              )
        error_list.append(error)
        mean_update_times.append(mean_update_time)

    print(f'optimizer: {args.optimizer}, const U: {args.const_u}, qr_every: {args.qr_every}, dataset: {args.dataset_name}\n'
          f'error mean: {np.mean(error_list)}, '
          f'error std: {np.std(error_list)}, '
          f'update time mean: {np.mean(mean_update_times)}')
