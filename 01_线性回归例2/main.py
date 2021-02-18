import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('dataset/education-income.csv')

# 读取X和Y
X = torch.from_numpy(data.Education.values.reshape(-1,1).astype(np.float32))
Y = torch.from_numpy(data.Income.values.reshape(-1,1).astype(np.float32))

# 初始化模型
w = torch.randn(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
learning_rate = 0.0001    # 学习速率

# 训练模型
for epoch in range(5000):
    for x,y in zip(X,Y):
        y_pred = torch.matmul(w,x) + b    # 模型预测
        loss = (y - y_pred).pow(2).mean()    # 损失函数（均方误差：差的平方再取均值）
        if not w.grad is None:    # grad的更新是累加的,所以需要重置
            w.grad.data.zero_()
        if not b.grad is None:
            b.grad.data.zero_()
        loss.backward()
        with torch.no_grad():    # 模型修正（更新w和b，注意这里不需要跟踪梯度）
            w.data -= learning_rate*w.grad.data
            b.data -= learning_rate*b.grad.data

# 打印结果
print('after 5000 iterates...\n')
print('w = ',w)
print('b = ',b)

# 画图观察
plt.scatter(data.Education,data.Income)
plt.plot(X.numpy(),(w*X+b).data.numpy(),c = 'r')    # 注意对(w*X+b)取出值再转ndarray，因为此时requires_grad还是True
plt.show()


