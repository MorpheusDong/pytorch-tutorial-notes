# 使用直线 y=kx+b 拟合点集

import numpy as np

# 根据给定的k和b计算点集的误差平方和
def compute_error_for_line_given_points(b, k, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (k * x + b)) ** 2    # 计算总的误差平方和
    return totalError / float(len(points))    # 取平均

# 梯度下降
def step_gradient(b_current, k_current, points, learningRate):
    b_gradient = 0
    k_gradient = 0
    N = float(len(points))    # 用于取平均
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((k_current * x) + b_current))    # 计算b的梯度
        k_gradient += -(2/N) * x * (y - ((k_current * x) + b_current))    # 计算k的梯度
    new_b = b_current - (learningRate * b_gradient)    # 梯度更新
    new_k = k_current - (learningRate * k_gradient)
    return [new_b, new_k]

# 梯度下降迭代
def gradient_descent_runner(points, starting_b, starting_k, learning_rate, num_iterations):
    b = starting_b
    k = starting_k
    for i in range(num_iterations):
        b, k = step_gradient(b, k, np.array(points), learning_rate)
    return [b, k]

# 主函数
def run():
    # 初始化
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001    # 学习速率
    initial_b = 0
    initial_k = 0
    num_iterations = 1000    # 迭代1000次

    # 打印对于初始k和b的误差
    print("Starting gradient descent at b = {0}, k = {1}, error = {2}"
          .format(initial_b, initial_k,
                  compute_error_for_line_given_points(initial_b, initial_k, points))
          )

    # 开始迭代
    print("Running...")
    [b, k] = gradient_descent_runner(points, initial_b, initial_k, learning_rate, num_iterations)

    # 迭代结束，打印结果
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, k,
                 compute_error_for_line_given_points(b, k, points))
          )

if __name__ == '__main__':
    run()