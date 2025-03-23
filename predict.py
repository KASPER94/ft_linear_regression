import argparse
import sys

def error(string):
		print(string)
		sys.exit(0)

def check_float(x):
	try:
		x = float(x)
	except:
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
		
def parse_plot(args):
	pass

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('mileage', type=int, help='mileage to predict')
	parser.add_argument('file', nargs='?', type=str, help='text file for input', default=None)
	# Bonus Part
	parser.add_argument('-sc', '--scatter', help='data scatter plot', type=str)
	args = parser.parse_args()

	try:
		theta = parse_file(args)
	except:
		theta = [0, 0]
	print(f"theta0 = {theta[0]}, theta1 = {theta[1]}")
	x = args.mileage
	x = check_float(x)
	# Y is the prediction so the price
	y = theta[0] + theta[1] * x
	print(f'ft_linear_regression prediction for {x} km: {y} $')

if __name__ == "__main__":
	main()