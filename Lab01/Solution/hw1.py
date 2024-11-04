import torch, torchvision
import matplotlib.pyplot as plt, math, random
import numpy as np, tqdm, pickle as pkl

def sigmoid(x: torch.Tensor):
	return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))

class PerceptronLayer:
	def __init__(self, ins: int, neurons: int, device='cpu'):
		self.ins = ins
		self.act = sigmoid
		self.derivate = sigmoid_derivative
		self.weights = torch.rand((ins, neurons), device=device)
		self.biases = torch.rand((neurons,), device=device)
	def forward(self, ins: torch.Tensor) -> torch.Tensor:
		self.last_ins = ins
		# print(ins.shape, self.weights.shape)
		return self.act((torch.mm(ins, self.weights) + self.biases))
	def backward(self, loss):
		print(loss.shape, self.derivate(self.last_ins).shape, self.last_ins.shape)
		self.errors = torch.mm(loss.t(), self.derivate(self.last_ins)).t()
		print(self.errors.shape, self.last_ins.shape)
		self.weights -= torch.mm(self.last_ins, self.errors) * LR
		self.biases -= self.errors * LR
		return self.errors



class NN:
	def __init__(self, layers: list[PerceptronLayer]):
		self.layers = layers
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self.last_ins = []
		for layer in self.layers:
			self.last_ins.append(layer.forward(x))
			x = self.last_ins[-1]
		return x
	def backward(self, loss: torch.Tensor):
		loss_temp = loss
		for layer in reversed(self.layers):
			# print(loss_temp.shape, layer.weights.shape, layer.last_ins[-1].shape)
			loss_temp = layer.backward(loss_temp)

def get_choices(x: torch.Tensor):
	# print(x[:10])
	# print(torch.argmax(x, dim=1)[:10])
	return torch.argmax(x, dim=1)

def train(nn: NN):
	for _ in tqdm.trange(EPOCHS):
		for i in tqdm.trange(TRAIN_SPLIT):
			(xxs, yys) = (xs[i*BATCH_SIZE:(i+1)*BATCH_SIZE], ys[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
			xxs = torch.stack([torchvision.transforms.functional.pil_to_tensor(x).reshape(28*28).to(dtype=torch.float, device=DEVICE) for x in xxs])
			yys = torch.tensor([[1.0 if i == y else 0.0 for i in range(10)] for y in yys], dtype=torch.float, device=DEVICE)
			y_pred = nn.forward(xxs)
			l = LOSS(y_pred, yys, reduction='none').reshape((BATCH_SIZE, 1))
			nn.backward(l)

def test(nn: NN):
	(xxs, yys) = (xs[TRAIN_SPLIT*BATCH_SIZE:], ys[TRAIN_SPLIT*BATCH_SIZE:])
	xxs = torch.stack([torchvision.transforms.functional.pil_to_tensor(x).reshape(28*28).to(dtype=torch.float, device=DEVICE) for x in xxs])
	yys = torch.tensor(yys, dtype=torch.float, device=DEVICE)
	y_pred = nn.forward(xxs)
	ls = torch.eq(get_choices(y_pred), yys)
	print("Test accuracy: ", sum(ls) / len(ls))
	return sum(ls) / len(ls)



if __name__ == '__main__':
	MNIST = torchvision.datasets.MNIST('carn/mnist-train/')
	xs, ys = zip(*MNIST)
	LR = 0.01
	EPOCHS = 1
	BATCH_SIZE = 1_000
	TRAIN_SPLIT = int((len(MNIST) // BATCH_SIZE) * 0.9)
	DEVICE = 'cpu'
	LOSS = torch.nn.functional.cross_entropy
	nn = NN([PerceptronLayer(784, 100, DEVICE), PerceptronLayer(100, 10, DEVICE)])
	print(MNIST)
	train(nn)
	pkl.dump(nn, open('nn.pt', 'wb'))
	nn = pkl.load(open('nn.pt', 'rb'))
	test(nn)
