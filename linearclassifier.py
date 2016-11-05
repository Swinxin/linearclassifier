# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 10:07:08 2016

@author: Administrator
"""
from pylab import *
import math
class Data:
    """
    创建一个对象，两个属性
    data：样本的特征
    target:样本的标签
    """
    def __init__(self,row,allnum=False):
        if allnum:
            self.data = map(float,row[:-1])
        else:
            self.data = row[:-1]
        self.target = int(row[-1])
def loadData(f,allnum=False):
    """
    加载数据集
    return 存放Data对象的list
    """
    rows = []
    for line in open(f):
        rows.append(Data(line.split(","),allnum))
    return rows

def plotDatas(rows):
    """
    绘制数据分布
    """
    
    xdm,ydm = [r.data[0] for r in rows if r.target == 1],\
              [r.data[1] for r in rows if r.target == 1]
    xdn,ydn = [r.data[0] for r in rows if r.target == 0],\
              [r.data[1] for r in rows if r.target == 0]
    plot(xdm,ydm,'go')
    plot(xdn,ydn,'ro')
    show()
def train(rows):
    """
    训练线性分类器
    """
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
def dotProduct(v1,v2):
    """
    内积
    """
    return sum([v1[i] * v2[i] for i in range(len(v1))])
    
def classify(sample,avgs):
    """
    可以判断sample到两个中心点（C0，C1）的距离判断sample属于哪个类别
    在此处使用内积方法；
    sign((x-(C0+C1)/2)*(C0-C1))
    展开公式sign(xC0-xC1-(C0**2-C1**2)/2)
    """
    b = (dotProduct(avgs[1],avgs[1]) - dotProduct(avgs[0],avgs[0]))/2
    y = dotProduct(sample,avgs[0]) - dotProduct(sample,avgs[1]) + b
    
    if y > 0:
        return 0
    else:
        return 1
#-----------features preprocess--------
def yesno(v):
    """
    取值为yes,no的维度的特征处理
    """
    if v == 'yes':
        return 1
    elif v == 'no':
        return -1
    else:
        return 0
def interest(i1,i2):
    """
    :处理兴趣特征,将标称型的特征转化为数字
    这样做是由缺点的，比如A喜欢篮球，B喜欢足球，这样AB的交集就是0
    但是足球和篮球都是运动类，都是球类，可以分层次给予0.8的打分。而不是0或1
    """
    l1 = i1.split(":")
    l2 = i2.split(":")
    x = 0
    for v in l1:
        if v in l2:
            x += 1
    return x 
def distance(d1,d2):
    """
    ：处理位置特征
    可以根据用户的地理位置，计算其距离，比如调用地图API计算距离
    这里不错处理
    """
    return 0
def postPreprocessing():
    """
    使用各个特征处理的方法，处理数据集
    """
    oldRows = loadData("matchmaker.csv")
    newRows = []
    for row in oldRows:
        d = row.data
        data = [float(d[0]),yesno(d[1]),yesno(d[2]),\
                float(d[5]),yesno(d[6]),yesno(d[7]),\
                interest(d[3],d[8]),row.target] #少了一项distance(d[4],d[9])
        newRows.append(Data(data))
    return newRows
    
def scala(rows):   
    """
    对数据归一化
    """
    #[(min,max),(),()]
    ranges = [(min([row.data[i] for row in rows]),max([row.data[i] for row in rows]))
            for i in range(len(rows[0].data))]
    #(x - min) / (max - min)
    #传入的d是一个list
    scalaFun = lambda d:[(d[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]) for i in range(len(ranges))]
    newrows = [Data(scalaFun (row.data) + [row.target]) for row in rows]
    return newrows, scalaFun 
def rbf(v1,v2,gamma=20):
    """
    kernel function
    """
    m = sum([(v1[i] - v2[i])**2 for i in range(len(v1))])
    return math.exp(-gamma * m)
def classify_kernel(sample,rows,offset,gamma = 10):
    """
    使用核函数
    """
    sum0 = 0.0
    sum1 = 0.0
    count0 = .0
    count1 = .0
    for row in rows:
        if row.target == 0:
            sum0 += rbf(sample,row.data,gamma)#在计算内积的时候用核函数替代
            count0 += 1
        else:
            sum1 += rbf(sample,row.data,gamma)
            count1 += 1
    y = (1.0/count0) *sum0 - (1.0/count1) *sum1 + offset
    if y < 0:
        return 0
    else:
        return 1
def offset(rows,gamma=10):
    """
    同一类别 样本和样本之间计算
    """
    l0 = []
    l1 = []
    for row in rows:
        if row.target == 0:
            l0.append(row.data)
        else:
            l1.append(row.data)
    sum0 = sum(sum([rbf(v1,v2,gamma) for v1 in l0]) for v2 in l0) 
    sum1 = sum(sum([rbf(v1,v2,gamma) for v1 in l1]) for v2 in l1) 
    return (1.0 / len(l1)**2)*sum1 - (1.0 / len(l0)**2)*sum0
if __name__ == "__main__":
    rows = loadData("agesonly.csv",True)
    avgs = train(rows)
    print classify([30,23],avgs)
    newrows = postPreprocessing()
    scalarows,fun =  scala(newrows)
    scala_avgs = train(scalarows)
    print "原始数据",newrows[0].data
    print "归一化后的数据",scalarows[0].data
    print  newrows[11].target,classify(fun(newrows[11].data),scala_avgs)
    