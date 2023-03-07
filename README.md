# Vectorize scaler function through broadcasting
##### 利用numpy.array类和torch.tensor类的广播机制，把标量函数改造为张量函数。  
标量函数是输入一个或多个自变量，输出一个标量函数值：   
$$
y=f(x_1, x_2,\cdots)
$$   
Method-1: 为了并行计算多种输入情况，可以采用```Python```自带的```map```函数   
```python
def add_two(x1, x2):
    return x1 + x2 

a = [1, 2]
b = [3, 4]
iterator = map(add_two, a, b)
res = list(iterator)
print(res)
```  

```map```还能实现类似数组广播的功能   

```python
def add_two(x1, x2):
    return x1 + x2 

a = [1, 2, 3, 4, 5]
iterator = map(lambda x: add2(x, 3), b)
res = list(iterator)
print(res)
```

Method-1: 根据张量网络 (tensor network) 的知识，每个自变量 $x_i$ 的多种取值，就相当于控制变量法，多个自变量的多种取值就相当于对结果扩维，类似于张量积 (tensor product)。下面用概率张量来举例并行计算的张量元素：   
![概率张量](/pic/probability_tensor.svg)  
概率张量可以展开成张量元素表：  
![标量列表](/pic/prob_table.svg)   
下面就可以利用```numpy.array```或```torch.tensor```类的自动广播功能来把标量结果列表改造为张量结果。  
$$
Y^{i,j,k} = F(x_1^i, x_2^j, x_3^k)
$$  
触发自动广播的核心代码为   
```python
x1 = x1.reshape(-1, 1, 1)
x2 = x2.reshape(1, -1, 1)
x3 = x3.reshape(1, 1, -1)
```