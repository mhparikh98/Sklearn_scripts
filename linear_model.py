import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error


X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.displot(y, bins=30)
# plt.show()

model = joblib.load("boston_linear_model.pkl")

# r2_score (R2 score or coefficient of determination) = 1 - SSr/SSt 
# SSr =  np.sum((y_pred - y_actual)**2), SSt = np.sum((y_actual - np.mean(y_actual))**2)
# r2_score(y_test, y_pred)

# mean_squared_error = mean_squared_error(y_test, y_pred)
# mse = np.sum((y_pred - y_actual)**2)

# root mean_squared_error = mean_squared_error(y_test, y_pred, squared=False)
# rmse = np.sqrt(mse/m)