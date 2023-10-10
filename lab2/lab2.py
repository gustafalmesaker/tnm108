import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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

#POLYNOMIAL BASIS FUNCTION
#convert one-dimensional array into three-dimensional
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
#print(poly.fit_transform(x[:, None]))
poly.fit_transform(x[:, None])


#pipeline to plug higher-dimensional data representation into a linear regression
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())

#sin wave with noise
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50) 
xfit = np.linspace(0, 10, 1000) 
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show() 


#GAUSSIAN BASIS FUNCTION
class GaussianFeatures(BaseEstimator, TransformerMixin):
    "Uniformly spaced Gaussian features for one-dimensional input"

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)


gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)
plt.show()

#REGULARIZATION
#(overfitting)
model = make_pipeline(GaussianFeatures(30), LinearRegression())
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis])) 
plt.xlim(0, 10)
plt.ylim(-4.5, 1.5)
plt.show()

def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)
    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_) 
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10)) 
    plt.show()

model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)

#RIDGE REGRESSION (L2 REGULARIZATION)
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')

#LASSO REGRESSION (L1 REG.)
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')