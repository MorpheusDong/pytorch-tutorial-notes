import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

# 读取数据
data = pd.read_csv('dataset/credit-a.csv', header=None)    # 不把第一行认作表头

# 数据预处理
X = data.iloc[:, :-1]    # 输入为前15列
Y = data.iloc[:, -1].replace(-1,0)    # 输出为第16列
X = torch.from_numpy(X.values).type(torch.float32)    # 转成tensor
Y = torch.from_numpy(Y.values.reshape(-1, 1)).type(torch.float32)    # Y是单一行的ndarray，要打平成653*1
# print(X.shape)
# print(Y.shape)

# 使用nn模块创建模型
model = nn.Sequential(
    nn.Linear(15,1),    # 第一层为线性模型
    nn.Sigmoid()    # 第二层映射到Sigmoid函数
)

loss_fn = nn.BCELoss()    # 损失函数：二元交叉熵
opt = torch.optim.Adam(model.parameters(), lr=0.001)    # 创建优化器

batches = 16    # 每批处理16组数据
num_of_batches = 653 // 16
epochs = 1000    # 迭代1000次

# 训练模型
for epoch in range(epochs):
    for i in range(num_of_batches):
        start = i*batches
        end = start+batches
        x = X[start:end]    # 切片取数据
        y = Y[start:end]
        y_pred = model(x)    # 计算预测值
        loss = loss_fn(y_pred,y)    # 计算损失
        opt.zero_grad()    # 清空梯度
        loss.backward()    # 计算梯度
        opt.step()    # 模型优化

# print(model.state_dict())    # 打印训练完的模型的参数（15个权重+1个偏移）

# 打印结果（模型预测的分类）
result = (model(X).data.numpy() > 0.5).astype('int')    # 把ndarray的值先用0.5筛成true或false，然后转整型变成0或1
# print(result)

# 查看模型预测的正确率
print((result == Y.numpy()).mean())    # 求均值即可
