import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import gradient_descent as gd
import data_engineering as de
import math

## TESTING LOGISITIC REGRESSION GRADIENT DESCENT BY HAND
# data = pd.read_csv("SampleNBAFileStandardized.csv")
# log_reg = LogisticRegression()
# log_reg.fit(X=data.loc[:,["AwayRating","HomeRating"]],y=data.loc[:,"HomeWin"])
# print(log_reg.intercept_)
# print(log_reg.coef_)
# print(log_reg.score(X=data.loc[:,["AwayRating","HomeRating"]], y=data.loc[:,"HomeWin"]))
# print(log_reg.predict_proba(np.array([[1,-1],[-1,1]])))
# log_reg_model_params = gd.logistic_regression_model(X=data.loc[:,["AwayRating","HomeRating"]],y=data.loc[:,"HomeWin"])
# t = log_reg_model_params.dot(np.array([1,1,-1]))
# print(1/(1+math.exp(t)))
# print(1/(1+math.exp(-t)))

### TESTING THE LATENT VARIABLE GRADIENT DESCENT BY HAND
## Creating basic round robin scenario
indicators = np.array([[0,1],[2,3],[0,2],[1,3],[0,3],[1,2],
                       [1,0],[3,2],[2,0],[3,1],[3,0],[2,1]])
x = pd.DataFrame()
x["Intercept"] = pd.Series([1]*len(indicators))
x["AwayRating"] = pd.Series([1500]*len(indicators))
x["HomeRating"] = pd.Series([1500]*len(indicators))
z = np.array([-2.,-1.,1.,2.]).reshape((-1,1))
y = np.array([2,5,3,1,10,-3,-1,2,-2,-6,-1,5]).reshape((-1,1))
x = gd.replace_design_latent(x, indicators, z)

p_means = np.copy(z)
p_vars = np.array([2.]*len(z)).reshape((-1,1))

## Getting initial linear regression parameters
lm = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
lm.fit(X=x,y=y)
param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1,1))
print("Baseline Accuracy: %.5f" % lm.score(X=x, y=y))
print(lm.coef_)

# x["RatingInteraction"] = pd.Series(x["AwayRating"]*x["HomeRating"])
# lm.fit(X=x,y=y)
# print("With Interaction Accuracy: %.5f" % lm.score(X=x,y=y))
# print(lm.coef_)

## Testing gradient function, testing to see if improvement using same model parameters after one gradient step
# gradient = gd.margin_model_derivative_z(response=y, design_matrix=x, param_vector=param_vector, indicators=indicators,
#                              weights=np.ones(len(y)).reshape((-1,1)),z=z, prior_means=p_means, prior_vars=p_vars, MAP=False)
# print(gradient)
# new_z = z + 0.001 * gradient
# print(new_z)
# x = gd.replace_design_latent(x, indicators, new_z)
# print(lm.score(X=x, y=y))


## Testing the full gradient descent method
new_z = np.copy(z)
old_x = np.copy(x)
new_acc = 0
tol = 1e-07
change = tol + 1
iterations = 0
while change > tol:
    start_acc = lm.score(X=old_x,y=y)
    new_z = gd.latent_margin_gradient_descent(response=y, design_matrix=old_x, param_vector=param_vector, indicators=indicators,
                                                 weights=np.ones(len(y)).reshape((-1,1)),z=new_z, prior_means=p_means, prior_vars=p_vars, MAP=False, show=False)
    new_x = de.replace_design_latent(design_matrix=old_x, indicators=indicators, z=new_z)
    finish_r2 = lm.score(X=new_x, y=y) # For internal checks to make sure gradient descent improving model
    lm.fit(X=new_x, y=y)
    param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))
    new_acc = lm.score(X=new_x, y=y)
    old_x = np.copy(new_x)
    change = new_acc - start_acc
    iterations += 1
print("Finished MLE with %d iterations" % iterations)
final_acc_MLE = new_acc
final_z_MLE = np.copy(new_z)

print("-------------------------------")

## Testing the full gradient descent method with MAP estimate
new_z = np.copy(z)
change = tol + 1
lm = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
lm.fit(X=x,y=y)
while change > tol:
    start_acc = lm.score(X=old_x,y=y)
    new_z = gd.latent_margin_gradient_descent(response=y, design_matrix=old_x, param_vector=param_vector, indicators=indicators,
                                                 weights=np.ones(len(y)).reshape((-1,1)),z=new_z, prior_means=p_means, prior_vars=p_vars, MAP=True, show=False)
    new_x = de.replace_design_latent(design_matrix=old_x, indicators=indicators, z=new_z)
    finish_acc = lm.score(X=new_x, y=y) # For internal checks to make sure gradient descent improving model
    lm.fit(X=new_x, y=y)
    param_vector = np.append(arr=lm.coef_, values=np.std(y)).reshape((-1, 1))
    new_acc = lm.score(X=new_x, y=y)
    old_x = np.copy(new_x)
    change = new_acc - start_acc
print("Finished MAP with %d iterations" % iterations)
final_acc_MAP = new_acc
final_z_MAP = np.copy(new_z)

print(final_acc_MLE)
print(final_z_MLE)
print(final_acc_MAP)
print(final_z_MAP)

