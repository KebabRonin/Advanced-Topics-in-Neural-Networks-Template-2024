import json, importlib, tqdm

cfg = json.load(open('config.json', 'r'))

for m in cfg['imports']:
	try:
		exec(f'{m} = importlib.import_module("{m}")')
	except:
		exec(f'{m} = importlib.import_module("{m}", ".")')

import torch, torchvision.transforms as transforms, torchvision, torch.utils.data
import wandb

wandb.init(project="Advanced-Topics-in-Neural-Networks-Template-2024", entity="hw3", config=cfg)

DEVICE = torch.device(cfg['device'])

if cfg['dataset'] == 'CIFAR-10':
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
elif cfg['dataset'] == 'CIFAR-100':
	trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
	testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
elif cfg['dataset'] == 'MNIST':
	trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
	testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True)

exec(f"model = {cfg['model']}.to(DEVICE)")

if cfg['optimizer'] == 'Adam':
	optimizer = torch.optim.Adam(model.parameters(), **cfg['optimizer_params'])
elif cfg['optimizer'] == 'AdamW':
	optimizer = torch.optim.AdamW(model.parameters(), **cfg['optimizer_params'])
elif cfg['optimizer'] == 'RMSprop':
	optimizer = torch.optim.RMSprop(model.parameters(), **cfg['optimizer_params'])
elif cfg['optimizer'] == 'SGD':
	optimizer = torch.optim.SGD(model.parameters(), **cfg['optimizer_params'])

if cfg['lr_scheduler'] == 'ReduceLROnPlateau':
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg['lr_scheduler_params'])
elif cfg['lr_scheduler'] == 'StepLR':
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['lr_scheduler_params'])
else:
	scheduler = None

exec(f'loss_fn = {cfg["loss"]}')

if cfg['early_stopping'] is not None:
	# early_stopping = torch.
	pass

## Training loop
for epoch in tqdm.trange(cfg['epochs']):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs.to(DEVICE))
		loss = loss_fn(outputs, labels.to(DEVICE))
		loss.backward()
		optimizer.step()

		# log statistics to wandb
		wandb.log({"loss": loss.item()})



wandb.finish()