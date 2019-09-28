import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./MLF_GP1_CreditScore.csv")

strong_corr_index = []

for i in range(0, 27):
	for j in range(i+1, 27):
		 # and abs(data.iat[i,j]) < 0.9
		if abs(data.corr().iat[i,j]) > 0.5:
			strong_corr_index.append((i,j))
			if data.corr().iat[i,j] > 0.9:
				print(data.columns[i])
				print(data.columns[j])
				print(data.corr().iat[i,j])
# for i,j in strong_corr_index:
# 	data.plot(kind="scatter", x=data.columns[i], y=data.columns[j])


# cor_mat = data.corr()
# plt.pcolor(cor_mat)

# for i in range(0, 9):
# 	data.hist(bins=10, column=[data.columns[j] for j in (3*i, 3*i+1, 3*i+2)])

# plt.show()

# for i in range(0, 27):
# 	print(data.columns[i])
# 	column = data[data.columns[i]]
# 	print("Mean: " + str(column.mean()))
# 	print("Median: " + str(column.median()))
# 	print("Standard Deviation: " + str(column.std()))
# 	print("Skewness: " + str(column.skew()))
# 	print("Kurtosis: " + str(column.kurt()))
# 	print("\n\n")