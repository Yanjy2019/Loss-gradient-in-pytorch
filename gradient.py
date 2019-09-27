import torch   #导入pytorch模块
import torch.nn.functional as F   #导入pytorch函数模块
x=torch.ones(1)                    #定义初始变量
print(x)
w=torch.tensor([2.],requires_grad=True)          #定义初始的w的值,并且需要定义它需要求取梯度
#w=torch.full([1],2) w.requires_grad_()这与上一句等效的，规定其可以求取梯度的属性
loss=F.mse_loss(torch.ones(1),x*w)               #采用默认的函数mse_loss函数
grad=torch.autograd.grad(loss,[w])               #自动求取函数关于变量的t梯度值
print(grad)
#loss.backward()
#print(w.grad)                            #与上自动求取梯度的API功能是一样的，只是用法略有不同

#训练pytorch中的softmax函数
a=torch.rand(3)
a.requires_grad_()
print(a)
p=F.softmax(a,dim=0)
print(torch.autograd.grad(p[1],[a],retain_graph=True))    #求取梯度和后面使用backward是一样的
p[1].backward(retain_graph=True)   #保持不变
print(a.grad)

#单层感知机模型的梯度计算推导实例
x=torch.randn(1,10)
w=torch.randn(1,10,requires_grad=True)
o=torch.sigmoid(x@w.t())
loss=F.mse_loss(torch.ones(1,1),o)
print(torch.autograd.grad(loss,w,retain_graph=True))
loss.backward()  #与上面是等效的
print(w.grad)

#多输出连接层的梯度计算推导公式
x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=True)
o=torch.sigmoid(x@w.t())
loss=F.mse_loss(torch.ones(1,2),o)
print(torch.autograd.grad(loss,w,retain_graph=True))  #自动计算梯度的函数API
loss.backward()
print(w.grad)

#链式法则的应用
x=torch.tensor(1.)
w1=torch.tensor(2.,requires_grad=True)
b1=torch.tensor(1.)
w2=torch.tensor(2.,requires_grad=True)
b2=torch.tensor(1.)
y1=w1*x+b1
y2=w2*y1+b2
g1=torch.autograd.grad(y2,[y1],retain_graph=True)[0]
g2=torch.autograd.grad(y1,[w1],retain_graph=True)[0]
g3=torch.autograd.grad(y2,[w1],retain_graph=True)[0]
print(g1*g2)
print(g3)

#反向传播过程解析
#2D函数最优化曲线
#画3D函数图像输出
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d
figure=plt.figure()
#ax = Axes3D(figure)
ax=figure.gca(projection="3d")
x1=np.linspace(-6,6,1000)
y1=np.linspace(-6,6,1000)
x,y =np.meshgrid(x1,y1)
z=(x**2+y-11)**2+(x+y**2-7)**2
#ax.plot_surface(x,y,z,rstride=10,cstride=4,cmap=cm.YlGnBu_r)
ax.plot_surface(x,y,z,cmap="rainbow")
plt.show()

#梯度下降法寻找2D函数最优值函数
def f(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
x=torch.tensor([-4.,0.],requires_grad=True)  #
optimizer=torch.optim.Adam([x],lr=1e-3)
for step in range(20000):
    pre=f(x)
    optimizer.zero_grad()
    pre.backward()
    optimizer.step()
    if step % 2000==0:
        print(step,x.tolist(),pre)


