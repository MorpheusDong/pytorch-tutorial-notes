# pytorch-tutorial-notes

### 开发环境配置

(1) AnaConda

(2) NVIDIA CUDA驱动

(3) Pytorch

(4) python IDE

以上自行搜索安装，速度不够选择国内镜像源或者挂代理。

安装pytorch
'''
conda install pytorch torchvision cudatoolkit=10.0
'''

### 问题

(1) pytorch安装中提示网络错误
重新执行一遍上面的命令就行。可以挂代理，速度快很多。

(2) 排查GPU是否可以使用：
https://blog.csdn.net/weixin_43981229/article/details/106673581 （检查显卡驱动的版本问题）
https://www.cnblogs.com/zhouzhiyao/p/11827267.html 
