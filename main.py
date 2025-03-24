import argparse
import sys
import os
import random
from utils import *

def main():
	parser = argparse.ArgumentParser(description="ft_linear_regression")
	parser.add_argument('--train', action='store_true', help="Tain the model")
	parser.add_argument('--predict', action='store_true', help="Model predict prices")
	parser.add_argument('--plot', action='store_true', help="Visualisation of the model")
	parser.add_argument('--eval', action='store_true', help="Model's evaluation")
	parser.add_argument('--compare', action='store_true', help="Model's compare with sklearn")
	parser.add_argument('--cd', action='store_true', help="Model's compare with sklearn")

	args = parser.parse_args()
	i = 0

	if not any(vars(args).values()):
		print("No args: Exec train -> predict")
		args.train = True
		args.predict = True
		i = random.randint(10000, 150000)
	
	if args.train:
		print("🧠 Train Model...")
		os.system('python3 train.py')

	if args.predict:
		print("🔮 Model makes prediction...")
		if i > 0:
			os.system(f'python3 predict.py {i} theta.txt')
		else:
			print('Enter the km of the car:', end='', flush=True)
			line = sys.stdin.readline().strip()
			os.system(f'python3 predict.py {line} theta.txt')

	if args.plot:
		print("📊 Display data...")
		os.system(f'python3 predict.py {i} theta.txt --sc')

	if args.eval:
		print("📈 Model's evaluation...")
		os.system('python3 eval.py')

	if args.compare:
		print("📈 Model compare with sklearn...")
		os.system('python3 train.py --compare')

	if args.cd:
		print('📈 Train Model with perfect / clean data set')
		os.system('python3 train.py --perfect')
		i = random.randint(10000, 150000)
		os.system(f'python3 predict.py {i} theta.txt --scatter --perfect')


if __name__ == "__main__":
	main()