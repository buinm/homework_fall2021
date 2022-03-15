import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV


# load data
dpa = pd.read_csv('./data/house-votes-84.complete.csv')
#dpa = pd.read_csv('./data/house-votes-simulated.complete.data')

dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
for i in range(16):
	index = 'A'+ str(i+1)
	dpa[index] = dpa[index].map({'y': 1, 'n': 0})
#dpa.info()

pay = dpa.Class
paX = dpa.drop('Class', axis = 1)

"""
  10-cv with house-votes-84.complete.csv using LASSO
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true
  NOTE you do *not* need to modify this function
"""
def lasso_evaluate(train_subset=False, subset_size = 0):
	sample_size = pay.shape[0]
	tot_incorrect=0
	tot_test=0
	tot_train_incorrect=0
	tot_train=0
	step = int( sample_size/ 10 + 1)
	ignored_rate_count = 0
	train_times = 0

	for holdout_round, i in enumerate(range(0, sample_size, step)):
		#print("CV round: %s." % (holdout_round + 1))
		if(i==0):
			X_train = paX.iloc[i+step:sample_size]
			y_train = pay.iloc[i+step:sample_size]
		else:
			X_train =paX.iloc[0:i]
			X_train = X_train.append(paX.iloc[i+step:sample_size], ignore_index=True)
			y_train = pay.iloc[0:i]
			y_train = y_train.append(pay.iloc[i+step:sample_size], ignore_index=True)

		X_test = paX.iloc[i: i+step]
		y_test = pay.iloc[i: i+step]

		if(train_subset):
			X_train = X_train.iloc[0:subset_size]
			y_train = y_train.iloc[0:subset_size]

		#print(" Samples={} test = {}".format(y_train.shape[0],y_test.shape[0]))
		# train the classifiers
		lasso = Lasso(alpha = 0.001)
		lasso.fit(X_train, y_train)
		train_times += 1

		# Q4
		# Check the parameters and count zeroed weight
		count_w = 0
		#print(lasso.coef_)
		for v in lasso.coef_:
			if abs(v) < 0.01:
				count_w += 1
		ignored_rate_count += count_w

		lasso_predit = lasso.predict(X_test)           # Use this model to predict the test data

		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0

		for (index, num) in enumerate(lasso_result):
			if(y_test.values.tolist()[index] != num):
				error+=1

		tot_train_incorrect+= error
		tot_train += len(lasso_result)

		#print('Error rate {}'.format(1.0*error/len(lasso_result)))
		lasso_predit = lasso.predict(X_train)           # Use this model to get the training error
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0

		for (index, num) in enumerate(lasso_result):
			if(y_train.values.tolist()[index] != num):
				error+=1

		#print('Train Error rate {}'.format(1.0*error/len(lasso_result)))
		tot_incorrect += error
		tot_test += len(lasso_result)
	#print('10CV Error rate {}'.format(1.0*tot_incorrect/tot_test))
	#print('10CV train Error rate {}'.format(1.0*tot_train_incorrect/tot_train))

	return 1.0*tot_incorrect/tot_test, 1.0*tot_train_incorrect/tot_train, ignored_rate_count / (16*train_times)

'''
 For Q5
 Run LASSO, using the commonly-used strategy of replacing unknown values with 0 (“no)
 We have modified the csv file for this purpose.
  NOTE you do *not* need to modify this function
  '''

def lasso_evaluate_incomplete_entry():
	# get incomplete data
	dpc = pd.read_csv('./data/house-votes-84.incomplete.csv')
	for i in range(16):
		index = 'A'+ str(i+1)
		dpc[index] = dpc[index].map({'y': 1, 'n': 0})
	lasso = Lasso(alpha = 0.001)
	lasso.fit(paX, pay)
	lasso_predit = lasso.predict(dpc)
	print(lasso_predit)


def main():
	'''
	TODO modify or use the following code to evaluate your implemented
	classifiers
	Suggestions on how to use the starter code for Q2, Q3, and Q5:

	#For Q2
	error_rate, unused = lasso_evaluate()
	print('10CV Error rate {}'.format(error_rate))
	#For Q3
	train_error = np.zeros(10)
	test_error = np.zeros(10)
	for i in range(10):
		x, y =lasso_evaluate(train_subset=True, subset_size=i*10+10)
		test_error[i] = y
		train_error[i] =x
	print(train_error)
	print(test_error)
	#Q4
	#TODO
	#You may find lasso.coef_ useful
	#Q5
	print('LASSO  P(C= 1|A_observed) ')
	lasso_evaluate_incomplete_entry()
	'''

	# Q2
	print("Q2 LASSO test\n")
	test_error, train_error, _ = lasso_evaluate()
	print('  10-fold cross validation total test error {:2.4f} and total train error '
	  '{:2.4f} '.format(test_error, train_error))


	# Q3
	x_axis = np.linspace(10, 100, 10)
	train_error = np.zeros(10)
	test_error = np.zeros(10)
	for x in range(10):
		data_size = (x+1) * 10
		test_err, train_err, _ = lasso_evaluate(train_subset=True, subset_size=data_size)
		train_error[x] = train_err
		test_error[x]  = test_err
		print("10-fold cross validation test error {:2.4f} and train error {:2.4f} using {} data train size".format(test_error[x], train_error[x], data_size))

	# Plotting test error
	plt.plot(x_axis, test_error, 'ro')
	plt.ylabel("Test error")
	plt.xlabel("Sample size")
	plt.show()

	# Plotting test error
	plt.plot(x_axis, train_error, 'ro')
	plt.ylabel("Train error")
	plt.xlabel("Sample size")
	plt.show()

	print(train_error)
	print(test_error)

	# Q4
	# TODO: Automatically change the data set to simulated data set

	x_axis = np.linspace(400, 4000, 10)
	ratio = np.zeros(10)
	for x in range(10):
		data_size = (x+1) * 400
		test_err, train_err, rat = lasso_evaluate(train_subset=True, subset_size=data_size)
		ratio[x] = rat
		print("10-fold cross validation non-partisan bills {:2.4f} using {} data train size".format(ratio[x], data_size))

	# Plotting test error
	plt.plot(x_axis, ratio, 'ro')
	plt.ylabel("Non-Partisan bills ratio")
	plt.xlabel("Sample size")
	plt.show()

	# Q5
	print('LASSO  P(C= 1|A_observed) ')
	lasso_evaluate_incomplete_entry()


if __name__ == "__main__":
	main()
