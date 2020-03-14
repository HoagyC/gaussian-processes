# Starting by following tutorial at https://peterroelants.github.io/posts/gaussian-process-tutorial/

import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt

# Draws {num_functions} selections at {num_samples} different points.
# {kernel} is a function which takes two inputs and returns their covariance.
def draw_gp_priors(num_samples, num_functions, kernel):
    xs = np.linspace(0, 10, num_samples)
    cov_matrix = np.array([[kernel(i, j) for j in xs] for i in xs])

    ys = np.random.multivariate_normal(mean=np.zeros(num_samples), cov=cov_matrix,
            size=num_functions)

    for i in range(num_functions):
        plt.plot(xs, ys[i], linestyle='-', marker='o', markersize=3)
    
    # Latex has a series of small spaces.
    # In order of increasing size:  \,    \:    \;
    # There's also a negative space: \!   - rogue!
    plt.xlabel('$sexy\:latex\:x$')
    plt.ylabel('normal y')
    plt.title(f'{num_functions} samples from prior')
    plt.gcf().canvas.set_window_title('hello!!')

    plt.show()

# Defining a load of common kernels
def constant_kernal(constant):
    def con_return(a, b):
        return constant
    return con_return 

brownian_kernel = min

# Note to self, not nearly as smooth if you accidentally use euclidean distance
# rather than squared euclidean distance.
def exp_quadratic(l=5):
    def exp_quadratic_return(a, b):
        sq_norm = -spatial.distance.sqeuclidean(a, b) / ((l**2) * 2)
        return np.exp(sq_norm)
    return exp_quadratic_return

def white_noise(sigma=1):
    def white_noise_return(a, b):
        return 1 if a == b else 0
    return white_noise_return

if __name__ == "__main__":
    draw_gp_priors(100, 10, exp_quadratic(0.5))

