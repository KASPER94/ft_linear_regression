import argparse
import sys
import matplotlib.pyplot as plt
from utils import process_csv_file


def error(string):
    print(string)
    sys.exit(0)


def check_float(x):
    try:
        x = float(x)
    except NameError:
        error("Wrong value")
    return (x)


def parse_file(args):
    with open(args.file) as file:
        if file.mode == 'r':
            theta = []
            lines = file.readlines()
            for raw in lines:
                theta.append(float(raw.split(':')[1]))
            return theta


def plot(x, y, theta0, theta1):
    y_pred = [theta0 + theta1 * i for i in x]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mileage', type=int,
                        help='mileage to predict')
    parser.add_argument('file', nargs='?', type=str,
                        help='text file for input', default=None)
    # Bonus Part
    parser.add_argument('-sc', '--scatter', action='store_true',
                        help='data scatter plot')
    parser.add_argument('--perfect', action='store_true',
                        help='clean / perfect data set')
    args = parser.parse_args()

    try:
        theta = parse_file(args)
    except NameError:
        theta = [0, 0]
    if args.scatter:
        if args.perfect:
            dataset = "perfect_data.csv"
        else:
            dataset = "data.csv"
        mileages, prices = process_csv_file(dataset)
        plot(mileages, prices, theta[0], theta[1])
    else:
        print(f"theta0 = {theta[0]}, theta1 = {theta[1]}")
        x = args.mileage
        x = check_float(x)
        # Y is the prediction so the price
        y = theta[0] + theta[1] * x
        if y < 0:
            print('ft_linear_regression prediction the car ')
            print('has too many km to be valuable')
        else:
            print(f'ft_linear_regression prediction for {x} km: {y} $')


if __name__ == "__main__":
    main()
