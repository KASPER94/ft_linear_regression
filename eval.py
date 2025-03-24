import csv
from math import sqrt

def load_data(filename):
    km = []
    price = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                km.append(float(row["km"]))
                price.append(float(row["price"]))
            except:
                continue
    return km, price

def load_theta(filename="theta.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()
        theta0 = float(lines[0].split(":")[1])
        theta1 = float(lines[1].split(":")[1])
    return theta0, theta1

def std_dev(values):
	m = len(values)
	mean = sum(values)
	return (sum((x - mean) ** 2 for x in values) / m) ** 0.5

def evaluate(km, price, theta0, theta1):
    m = len(km)
    y_pred = [theta0 + theta1 * x for x in km]

    mse = sum((y_pred[i] - price[i])**2 for i in range(m)) / m
    rmse = sqrt(mse)
    mae = sum(abs(y_pred[i] - price[i]) for i in range(m)) / m

    # R² : 1 - (erreur modèle / erreur moyenne)
    mean_y = sum(price) / m
    total_var = sum((price[i] - mean_y)**2 for i in range(m))
    residual_var = sum((price[i] - y_pred[i])**2 for i in range(m))
    r2 = 1 - (residual_var / total_var)

    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f} €")
    print(f"MAE  : {mae:.2f} €")
    print(f"R²   : {r2:.4f}")
    


def main():
    km, price = load_data("data.csv")
    theta0, theta1 = load_theta()
    evaluate(km, price, theta0, theta1)

if __name__ == "__main__":
    main()