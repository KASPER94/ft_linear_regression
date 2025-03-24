import csv
import os
import numpy as np

def read_file(file):
	try:
		data = open(file, "r")
		print(data)
	except BaseException:
		print("Error during data reading")
		exit()

def process_csv_file(filename):
	mileages = []
	prices = []
	with open(filename, "r") as file:
		reader = csv.DictReader(file)
		for row in reader:
			try:
				mileages.append(float(row["km"]))
				prices.append(float(row["price"]))
			except:
				continue
	return mileages, prices

def compare_with_sklearn(mine):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    mileages, prices = process_csv_file("data.csv")
    
    # Reshape pour scikit-learn
    X = np.array(mileages).reshape(-1, 1)
    y = np.array(prices)
    
    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X, y)
    
    # Coefficients
    print(f"scikit-learn - intercept: {model.intercept_}, coefficient: {model.coef_[0]}")
    print(f"ft_linear_regression - intercept: {mine.theta0}, coefficient: {mine.theta1}")
    
    # Prédiction pour 61789 km
    prediction = model.predict([[61789]])[0]
    print(f"scikit-learn prediction for 61789 km: {prediction} $")
    os.system('python3 predict.py 61789 theta.txt')
    
    # MSE
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"ft_linear_regression Mean Squared Error: {mine.mse}")