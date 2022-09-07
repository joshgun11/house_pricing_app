from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

class KModel():

    def __init__(self) -> None:
        pass

    def linear_regression(self):
        reg = LinearRegression()
        return reg

    def lasso_regresion(self,alpha):
        reg = Lasso(alpha)
        print(alpha)
        return reg

    def ridge_regresion(self,alpha):
        reg = Ridge(alpha)
        print(alpha)
        return reg

    def tree_regresion(self):
        reg = RandomForestRegressor()
        return reg

    