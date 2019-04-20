from temp import *

dataSet_train_file = open("./MNIST_data/mnist_train.csv", 'r')
dataSet_train = dataSet_train_file.readlines()
dataSet_train_file.close()
dataSet_test_file = open("./MNIST_data/mnist_test.csv", 'r')
dataSet_test = dataSet_test_file.readlines()
dataSet_test_file.close()


l = [784, 100, 10]
nn = neural_network(layouts=l, activation_func=sigmoid, target_func=mse)
nn.bias_init()
nn.weight_init(uniform=True, glorot=True)
nn.train_without_batch(dataSet_train)
nn.launch_test(dataSet_test)

