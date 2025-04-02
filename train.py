from utils import process_csv_file, compare_with_sklearn
import argparse


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true',
                        help='Compare model with sklearn')
    parser.add_argument('--perfect', action='store_true',
                        help='Compare model with sklearn')
    args = parser.parse_args()

    dataset = "perfect_data.csv" if args.perfect else "data.csv"
    mileages, prices = process_csv_file(dataset)
    model = linear_reg()
    model.fit(mileages, prices)
    # model.fit_analytically(mileages, prices)
    model.save()
    if args.compare:
        compare_with_sklearn(model)


if __name__ == "__main__":
    main()
