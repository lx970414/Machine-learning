'''
数据集：连续数据，二分类，最后一列是标签列
'''

import numpy as np

def loaddataset(filename):
	fp = open(filename)
	dataset = []
	labelset = []
	for i in fp.readlines():
		a = i.strip().split()

		#存储属性数据
		dataset.append([float(j) for j in a[:len(a)-1]])

		#存储标签数据
		labelset.append(int(float(a[-1])))
	return dataset, labelset

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def trainning(dataset, labelset, test_data, test_label):
	#将列表转化为矩阵
	data = np.mat(dataset)
	label = np.mat(labelset).transpose()

	#初始化参数w
	w = np.ones((len(dataset[0])+1, 1))

	#属性矩阵最后添加一列全1列（参数w中有常数参数）
	a = np.ones((len(dataset), 1))
	data = np.c_[data, a]

	#步长
	n = 0.0001

	#每次迭代计算一次正确率（在测试集上的正确率）
	#达到0.75的正确率，停止迭代
	rightrate = 0.0
	while rightrate < 0.75:
		#计算当前参数w下的预测值
		c = sigmoid(np.dot(data, w))

		#梯度下降的计算过程，对照着梯度下降的公式
		b = c - label
		change = np.dot(np.transpose(data), b)
		w = w - change * n

		#预测，更新正确率
		rightrate = test(test_data, test_label, w)
	return w

def test(dataset, labelset, w):
	data = np.mat(dataset)
	a = np.ones((len(dataset), 1))
	data = np.c_[data, a]

	#使用训练好的参数w进行计算
	y = sigmoid(np.dot(data, w))
	b, c = np.shape(y)

	#记录预测正确的个数，用于计算正确率
	rightcount = 0

	for i in range(b):

		#预测标签
		flag = -1

		#大于0.5的为正例
		if y[i, 0] > 0.5:
			flag = 1

		#小于等于0.5的为反例
		else:
			flag = 0

		#记录预测正确的个数
		if labelset[i] == flag:
			rightcount += 1

	#正确率
	rightrate = rightcount / len(dataset)
	return rightrate

if __name__ == '__main__':
	dataset, labelset = loaddataset('Training.txt')

	test_data, test_label = loaddataset('Test.txt')
	w = trainning(dataset, labelset, test_data, test_label)
	rightrate = test(test_data, test_label, w)
	print("Precision is:%f"%(rightrate))
