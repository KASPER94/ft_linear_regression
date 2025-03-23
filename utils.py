
def read_file(file):
	try:
		data = open(file, "r")
		print(data)
	except BaseException:
		print("Error during data reading")
		exit()