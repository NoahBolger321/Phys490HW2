from data_handling import MnistData


data = MnistData(test_size=0.1017)

X_train = data.X_train
Y_train = data.Y_train
X_test = data.X_test
Y_test = data.Y_test

