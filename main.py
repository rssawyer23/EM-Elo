import numpy as np
import gradient_descent as gd
from sklearn.linear_model import LinearRegression


def fit_margin_model(x, y, indicators, p_means, p_vars, MAP=False, show=False, tol=1e-07):
    """
    
    :param x: data matrix (n x d) with entries for team ratings (can default to 0s here
    :param y: vector of margin of victories for the games (n x 1)
    :param indicators: matrix with team identifiers as entries (n x 2) matrix
    :param p_means: vector of prior means of the latent team variables (z x 1)
    :param p_vars: vector of prior variances of the latent team variables (z x 1)
    :param MAP: boolean determining if MLE (False, default) should be used or the MAP estimate
    :param show: boolean determining whether additional information should be printed to the console
    :param tol: tolerance for convergence
    :return: final latent variables (z x 1) and final accuracy of the model (single float)
    """
    lm = LinearRegression()
    lm.fit(X=x, y=y)
    param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))

    change = tol + 1
    iterations = 0
    new_z, new_acc = 0, 0
    while change > tol:
        start_acc = lm.score(X=x, y=y)
        new_z, x = gd.latent_margin_gradient_descent(response=y, design_matrix=x, param_vector=param_vector,
                                                     indicators=indicators,
                                                     weights=np.ones(len(y)).reshape((-1, 1)), z=new_z,
                                                     prior_means=p_means, prior_vars=p_vars, MAP=MAP, show=show)
        finish_acc = lm.score(X=x, y=y)  # For internal checks to make sure gradient descent improving model
        lm.fit(X=x, y=y)
        param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))
        new_acc = lm.score(X=x, y=y)
        change = new_acc - start_acc
        iterations += 1
    print("Finished with %d iterations" % iterations)
    return new_z, new_acc