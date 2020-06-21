from network import MultilayerNeuralNetwork
import numpy as np
import mnist
import argparse
import sys


parser = argparse.ArgumentParser(description='Hand Written Digit Recognition.')
parser.add_argument('--train', help='--train model_name')
parser.add_argument('--batch_size',default=1000,type=int,
					help='--loadmodel model_name')
parser.add_argument('--epochs',default=100, type=int,
					help='--epochs 100')
parser.add_argument('--lr',default=0.01, type=float,
					help='--lr 0.1')
parser.add_argument('--loadmodel', help='--loadmodel model_name.npz')
parser.add_argument('--evaluate', help='--evaluate model_name.npz')

if not len(sys.argv) > 1:
	parser.print_help()
	sys.exit(0)
else:
	args = parser.parse_args()
	# exit(0)


def vectorize(num):
	vec = np.zeros((10, 1))
	vec[num] = 1.0

	return vec


# mnist.datasets_url = './MNIST_Dataset'
mnist.temporary_dir = lambda: './MNIST_Dataset'
train_images = mnist.train_images()
train_images = train_images.reshape(train_images.shape[0],-1).T
train_labels = mnist.train_labels()
train_labels = np.array([vectorize(num) for num in train_labels])
train_labels = train_labels.reshape(train_labels.shape[0],-1).T

test_images = mnist.test_images()
test_images = test_images.reshape(test_images.shape[0],-1).T
test_labels = mnist.test_labels()

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)
# print(train_labels[:,0])

network_dimension = [train_images.shape[0],300,10]

nn = MultilayerNeuralNetwork(layers_dims=network_dimension,
							layer_activation_func=["tanh","sigmoid"])

if args.evaluate is not None:
	nn.load_weights(args.evaluate)

if args.loadmodel != None:
	nn.load_weights(args.loadmodel)

if args.train is not None:
	nn.train(train_images, train_labels,
		args.batch_size, args.epochs,
		args.lr,args.train)

predictions = nn.predict(test_images)
predictions = np.argmax(predictions,axis=0)

correct = 0
for pred,actual in zip(predictions,test_labels):
	if pred == actual:
		correct += 1

accuracy = (correct/test_labels.shape[0])*100
print("Accuracy : %.2f %%"%(accuracy))