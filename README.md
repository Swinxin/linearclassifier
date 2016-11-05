# 总结
## 线性分类器，核方法
- 1.数据封装：将每一个样本封装为一个对象包含`data`和`target`，将整个数据集放入一个list中
```python
class Data:
    def __init__(self,row):
        self.data = map(float,row[:-1])
        self.target = int(row[-1])
```
- 2.基本的线性分类器：原理是寻找每个类别中所有数据的平均值，构造一个代表该类别的中心点，有新数据要对其进行分类时，只需要通过判断距离哪个中心点位置近in行分类。
```python
def train(rows):
    average = {}#用来存放不同类别的中心点
    counts = {}
    for row in rows:
        c1 = row.target
        average.setdefault(c1,[0.0]*len(row.data))
        counts.setdefault(c1,0)
        for i in range(len(row.data)):
            average[c1][i] += float(row.data[i])
        counts[c1] += 1
        
    for c,avg in average.items():
        for i in range(len(avg)):
            avg[i] /= counts[c]#average的值会修改
    return average
```
有一点值得注意，**average**key为类别，value为一个list，在修改value的时候，average的值也跟着变化
- 3.距离的计算
在对测试样本进行分类的时候需要计算样本与类别中心点的距离，可以使用**欧氏距离**，也可以使用向量的夹角
class = sign((X - (C1+C2)/2) * (C2 - C1)) ==>sign(xC1-XC2 + (C1*C1 - C2*C2)/2)
通过计算向量夹角判断样本的类别
- 4.特征的处理：针对每一特征维度，定制不同的特征处理方法，最后再形成新的数据集
    - 固定个数标称型特征，如Yes or No，可以化为1,0类型
    - 个数不固定的标称型特征，如足球，篮球，滑雪，看书等等，可以特征进行按层级排列，例如篮球和足球都属于球类，球类都属于运动，在转化为数值特征的时候，就可以加上0.6....而不是+1，
    - 在处理位置特征时候，可以借助地图API计算距离
最后的合并成新的数据集`[f1(row[0]),f2(row[1]),f3(row[2])]`
- 4.**对特征进行缩放**：（加黑了，说明很重要）通过找出特征的最大值和最小值，将数据都缩放到同一尺度
```python
def scala(rows):   
    #[(min,max),(),()]
    ranges = [(min([row.data[i] for row in rows]),max([row.data[i] for row in rows]))
            for i in range(len(rows[0].data))]
    #(x - min) / (max - min)
    scalaFun = lambda d:[(d[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]) for i in range(len(ranges))]
    newrows = [Data(scalaFun (row.data) + [row.target]) for row in rows]
    return newrows, scalaFun 
```
返回两部分，一个是处理后的数据集，一个是scala函数，用于处理新样本。至于为什么写成匿名函数，这样就能够保存`ranges`了
- 5.核方法：核技巧是用一个新的函数替代原来的点积函数，借助某个映射函数将数据变换到更高维度的坐标空间，新函数返回新的内积结果。
rbf核
```python
def rbf(v1,v2,gamma=20):
    m = sum([(v1[i] - v2[i])**2 for i in range(len(v1))])
    return math.exp(-gamma * m)
```

 

