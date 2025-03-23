from utils import *
import csv
import argparse
import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt

class linear_reg:
    def __init__(self, learning_rate=0.1, iterations=10000):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.mse = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
    
    def normalize_x(self, set_x):
        self.min_x = min(set_x)
        self.max_x = max(set_x)
        return [(x - self.min_x) / (self.max_x - self.min_x) for x in set_x]
    
    def normalize_y(self, set_y):
        self.min_y = min(set_y)
        self.max_y = max(set_y)
        return [(y - self.min_y) / (self.max_y - self.min_y) for y in set_y]
    
    def denormalize(self):
        x_scale = self.max_x - self.min_x
        y_scale = self.max_y - self.min_y
        
        self.theta1 = self.theta1 * (y_scale / x_scale)
        self.theta0 = self.theta0 * y_scale + self.min_y - (self.theta1 * self.min_x)
        
        self.mse = self.mse * 2 * (y_scale ** 2)
    
    def fit(self, x, y):
        m = len(x)
        y_normalized = self.normalize_y(y)
        x_normalized = self.normalize_x(x)
        previous_cost = float('inf')
        
        for _ in range(self.iterations):
            sum_error0 = 0
            sum_error1 = 0
            for i in range(m):
                predict = self.theta0 + self.theta1 * x_normalized[i]
                error = predict - y_normalized[i]
                sum_error0 += error
                sum_error1 += error * x_normalized[i]
            self.theta0 -= self.learning_rate * (1/m) * sum_error0
            self.theta1 -= self.learning_rate * (1/m) * sum_error1
            
            current_cost = 0
            
            for j in range(m):
                predict = self.theta0 + self.theta1 * x_normalized[j]
                current_cost += (predict - y_normalized[j])**2
            current_cost /= (2*m)

            if abs(previous_cost - current_cost) < 1e-9:
                break
            
            self.mse = previous_cost = current_cost
        
        self.denormalize()
    
    def save(self):
        with open("theta.txt", "w") as f:
            f.write(f"theta0: {self.theta0}\ntheta1: {self.theta1}")
	
    def fit_analytically(self, x, y):
        m = len(x)
        x_mean = sum(x) / m
        y_mean = sum(y) / m

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(m))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(m))

        self.theta1 = numerator / denominator
        self.theta0 = y_mean - self.theta1 * x_mean

        total_error = sum((self.theta0 + self.theta1 * x[i] - y[i])**2 for i in range(m))
        self.mse = total_error / m

    def plot(self, x, y):
        y_pred = [self.theta0 + self.theta1 * i for i in x]
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Données réelles')
        plt.plot(x, y_pred, color='red', label='Regression Linéaire')
        plt.xlabel("Kilométrage (km)")
        plt.ylabel("Prix (€)")
        plt.title("Prix estimé vs Kilométrage")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



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

def main():
	mileages, prices = process_csv_file("data.csv")
	model = linear_reg()
	model.fit(mileages, prices)
	# model.fit_analytically(mileages, prices)
	model.save()
	# compare_with_sklearn(model)
	model.plot(mileages, prices)

if __name__ == "__main__":
	main()