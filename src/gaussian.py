from src.createHistograms import all_values
import math
import numpy as np
from joblib import Parallel, cpu_count, delayed
import pickle


def run_gmm(key, geometry_dict):
    print(key)
    values = geometry_dict[key]
    gmm_results = gaussian_mixture_model(key, values)
    return gmm_results


def gauss(x, x0, sigma):
    """
    This function returns a Gaussian distribution.
    """
    return (1/(sigma*math.sqrt(2 * math.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2))


def gaussian_mixture_model(geometry_type, values):
    """
    GMM of one type
    """
    # peak_file = open("Dataset/Cyclic/list_peaks_qm9.txt", 'r')
    peak_file = open("list_peaks_mit.txt", 'r')
    # peak_file = open("Dataset/Cyclic/list_peaks_mit.txt", 'r')
    # peak_file = open("Dataset/Cyclic/list_peaks_EM_kaust.txt", 'r')
    for line in peak_file:
        line_split = line[:-1].split('\t')
        if geometry_type.__eq__(line_split[0]):
            num_peaks = int(line_split[1])
            break
    theta = []
    for it in range(num_peaks):
        mu_start = it/num_peaks*max(values)
        sd_start = 0.5
        theta.append([mu_start, sd_start])
    print(theta)
    theta_v = theta
    tol = 1e-5  # Tolerance value for convergence
    max_iter = 100
    trunc = 0.00001
    LL_old = -np.Infinity
    LL = []
    for i in range(max_iter):
        print("Iteration {}/{}".format(i+1, max_iter))
        matrix = []
        p_m = []
        for value in values:
            peak_g = np.array([])
            for j in range(num_peaks):
                mu = theta_v[j][0]
                sig = theta_v[j][1]
                w = float(gauss(value, mu, sig))
                peak_g = np.append(peak_g, w)
                # print("peak_g is {}".format(peak_g))
            p_m.append(peak_g)
            if np.sum(peak_g) == 0:
                peak_prob = np.zeros(num_peaks)
            else:
                peak_prob = peak_g / np.sum(peak_g)
            matrix.append(peak_prob)
        p_m = np.asarray(p_m)
        matrix = np.asarray(matrix).astype(np.float)
        N_values = np.sum(matrix, axis=0)
        p_values = np.log10(np.sum(p_m, axis=1))
        LL_new = np.sum(p_values)
        LL.append(LL_new)
        # print("The old LL is {} and the new LL is {}".format(LL_old, LL_new))
        print("The log-likelihood is {}".format(LL_new))
        if np.abs(np.abs(LL_old/LL_new) - 1) < tol:
            print("Converged!")
            break
        else:
            theta_v = []
            for k in range(num_peaks):
                mu = 1 / N_values[k] * matrix[:, k].dot(values)
                sig = max(math.sqrt((1 / N_values[k] * matrix[:, k].dot((values - mu) ** 2))), trunc)
                theta_v.append([mu, sig])
            theta_v = np.asarray(theta_v)
            theta_opt = theta_v
            LL_old = LL_new
    return theta_opt, LL


def gmm(data):
    print("Start calculating all geometry features.")
    geometry_dict, bad = all_values(data)
    for key in geometry_dict:
        print(key)
    ll_dict = {}
    gmm_dict = {}
    n_jobs = cpu_count()
    gmm_info = Parallel(n_jobs=n_jobs)(delayed(run_gmm)(keg, geometry_dict) for keg in geometry_dict)
    for key, i in zip(geometry_dict, range(len(geometry_dict))):
        gmm_dict[key] = gmm_info[i][0]
        ll_dict[key] = gmm_info[i][1]
    return gmm_dict, ll_dict, geometry_dict, bad


def peak_filtering(geometry_dict, decimals=3):
    new_geometry_dict = {}
    for key in geometry_dict:
        values = geometry_dict[key]
        new_geometry_dict[key] = np.round(values, decimals)
    return new_geometry_dict

