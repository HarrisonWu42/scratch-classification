"""
This file is used to define ResNet model and experiment.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


def get_confusion_matrix(epoch, y_true, y_pred, normalize=False, cmap=plt.cm.Blues):
	y_true_cm = []
	y_pred_cm = []
	for e in y_true:
		if e == 0:
			y_true_cm.append([1, 0, 0, 0, 0, 0])
		elif e == 1:
			y_true_cm.append([0, 1, 0, 0, 0, 0])
		elif e == 2:
			y_true_cm.append([0, 0, 1, 0, 0, 0])
		elif e == 3:
			y_true_cm.append([0, 0, 0, 1, 0, 0])
		elif e == 4:
			y_true_cm.append([0, 0, 0, 0, 1, 0])
	y_true_cm = np.array(y_true_cm)
	for e in y_pred:
		if e == 0:
			y_pred_cm.append([1, 0, 0, 0, 0, 0])
		elif e == 1:
			y_pred_cm.append([0, 1, 0, 0, 0, 0])
		elif e == 2:
			y_pred_cm.append([0, 0, 1, 0, 0, 0])
		elif e == 3:
			y_pred_cm.append([0, 0, 0, 1, 0, 0])
		elif e == 4:
			y_pred_cm.append([0, 0, 0, 0, 1, 0])
	y_pred_cm = np.array(y_pred_cm)
	y_true = y_true_cm.argmax(axis=1)
	y_pred = y_pred_cm.argmax(axis=1)

	y_true_clean = []
	y_pred_clean = []
	for i in range(len(y_true)):
		if y_true[i] > 1 and y_pred[i] > 1:
			y_true_clean.append(y_true[i])
			y_pred_clean.append(y_pred[i])

	classes = np.array(['0', '1', '2', '3', '4'])
	classes = classes[unique_labels(y_true_clean, y_pred_clean)]
	cm = metrics.confusion_matrix(y_true_clean, y_pred_clean)
	print(cm)
	# Only use the labels that appear in the data
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=classes, yticklabels=classes,
		   title="Image Classification",
		   ylabel='True label',
		   xlabel='Predicted label')

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	y = np.append(y_true_clean, y_pred_clean)
	plt.xlim(-0.5, len(np.unique(y)) - 0.5)
	plt.ylim(len(np.unique(y)) - 0.5, -0.5)
	plt.savefig('cm_image_classification' + str(epoch) + '.png')
	return cm


# labels
labelpath = '../data/info.csv'
data = pd.read_csv(labelpath)
labels = data['label'].values.tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
EPOCH = 20
lr = 0.001

imagePath = './images/'

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = resnet18(pretrained=False, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss().to(device)  # The loss function is cross entropy.
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
					  weight_decay=5e-4)  # The optimization method is mini-batch momentum-SGD, and uses L2 regularization (weight attenuation)

AllLabels = labels
lossList = []

for epoch in range(EPOCH):
	sum_loss = 0
	correct_train = 0
	total_train = 0

	images = []
	for index in tqdm(range(0, 20598)):
		fileName = str(index) + '_' + str(AllLabels[index]) + '.jpg'
		image = Image.open(imagePath + fileName)
		image = transform(image)
		image = torch.unsqueeze(image, 0)
		images.append(image)
	label = torch.from_numpy(np.array(AllLabels))
	trainimg = images[:int(len(images) * 0.8)]
	testimg = images[int(len(images) * 0.8):]
	trainlabel = label[:int(len(label) * 0.8)]
	testlabel = label[int(len(label) * 0.8):]
	trainset = TensorDataset(torch.cat(trainimg, 0), trainlabel)
	testset = TensorDataset(torch.cat(testimg, 0), testlabel)

	trainloader = DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=0)
	testloader = DataLoader(dataset=testset, batch_size=100, shuffle=True, num_workers=0)

	# train
	for step, (inputs, label) in tqdm(enumerate(trainloader)):
		length = len(trainloader)
		inputs, label = inputs.to(device), label.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, label)
		loss.backward()
		optimizer.step()
		print(loss.item())
		total_train += label.size(0)
		sum_loss += loss.item()
		lossList.append(loss.item())
		_, predicted = torch.max(outputs.data, 1)
		correct_train += predicted.eq(label.data).cpu().sum()
	print('[epoch: %d ] Loss: %.03f | Acc: %.3f%%' % (epoch + 1, sum_loss / 50000, 100.0 * correct_train / total_train))

	# test
	correct_test = 0
	total_test = 0
	label_pred = []
	label_true = []
	for step, (inputs, label) in tqdm(enumerate(testloader)):
		length = len(trainloader)
		inputs, label = inputs.to(device), label.to(device)
		model.eval()
		outputs = model(inputs)
		total_test += label.size(0)
		_, predicted = torch.max(outputs.data, 1)
		correct_test += predicted.eq(label.data).cpu().sum()
		label_pred = label_pred + predicted.tolist()
		label_true = label_true + label.tolist()
	print('[epoch: %d ] Acc: %.3f%%' % (epoch + 1, 100.0 * correct_test / total_test))

	accuracy = metrics.accuracy_score(label_true, label_pred)
	precision = metrics.precision_score(label_true, label_pred, average='weighted')
	recall = metrics.recall_score(label_true, label_pred, average='weighted')
	f1_micro = metrics.f1_score(label_true, label_pred, average='micro')
	f1_macro = metrics.f1_score(label_true, label_pred, average='macro')
	f1_weighted = metrics.f1_score(label_true, label_pred, average='weighted')
	print("accuracy:    ", accuracy)
	print("precision:   ", precision)
	print("recall:      ", recall)
	print("f1_micro:    ", f1_micro)
	print("f1_macro:    ", f1_macro)
	print("f1_weighted: ", f1_weighted)
	confusion_matrix = get_confusion_matrix(epoch, label_true, label_pred, normalize=True)
	print(confusion_matrix)
