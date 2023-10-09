import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#SIMPLE LINEAR REGRESSION
#Plot the points without line
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
#plt.scatter(x, y)
#plt.show()

#Points from before with linear reg
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

#Plottar scatter plot med linear reggression och printar Model slope/intercept
#plt.scatter(x, y)
#plt.plot(xfit, yfit)
#plt.show()
#print("Model slope: ", model.coef_[0])
#print("Model intercept:", model.intercept_)

#multidimensional linear models
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])
model.fit(X, y)
print(model.intercept_)
print(model.coef_)

#BASIS FUNCTIONS
#linear to none linear by adding polynom
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())


