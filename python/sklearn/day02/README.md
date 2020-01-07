- https://blog.csdn.net/Orange_Spotty_Cat/article/details/80520839
- https://blog.csdn.net/liweibin1994/article/details/79462554
- https://www.jianshu.com/p/c61ae11cc5f6

# roc曲线图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200102164155324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)

# 自定义ROC曲线与auc计算
## 混淆矩阵

![](https://img-blog.csdn.net/20180306192541902?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl3ZWliaW4xOTk0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


![](https://img-blog.csdn.net/20180531115939413?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09yYW5nZV9TcG90dHlfQ2F0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## F1 Score
![](https://img-blog.csdn.net/20180531141602829?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09yYW5nZV9TcG90dHlfQ2F0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

其中，P代表Precision，R代表Recall。

F1-Score指标综合了Precision与Recall的产出的结果。F1-Score的取值范围从0到1的，1代表模型的输出最好，0代表模型的输出结果最差。


## ROC曲线
ROC曲线是以 FPR = FP/(FP+TN)为横坐标，TPR = TP/(TP+FN)为纵坐标 （TPR也叫召唤率）

准确率定义为 P = TP/(TP+FP)

我们来举一个例子。比如我们有5个样本：

真实的类别(label)为y = c(1,1,0,0,1).

一个分类器预测样本为1的概率为p=c(0.5, 0.6, 0.55, 0.4, 0.7).

正如上面说的，我们需要有阈值，才能将概率转换为类别，才能得到FPR和TPR。而选定不同的阈值会得到不同的FPR和TPR。假设我们现在选定的阈值为0.1,那么5个样本都被归类为1。如果选定0.3，结果仍然一样。如果选了0.45作为阈值，那么只有样本4被分进0，其余都进入1类。当我们不断改变阈值，就会得到不同的FPR和TPR。然后我们将得到的(FPR , TPR)连接起来，就得到了ROC曲线了。

```python
"""
# 自定义roc曲线计算与auc计算
https://blog.csdn.net/liweibin1994/article/details/79462554
https://blog.csdn.net/Orange_Spotty_Cat/article/details/80520839
"""
from sklearn.metrics import roc_curve, auc,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 自定义roc曲线
def roc_curve_custom(y_true,y_proba):
    """难点：thresholds的确定"""
    # thresholds = np.linspace(0,1,5)
    # all possibly relevant bias parameters stored in a list
    thresholds = sorted(list(set(y_proba)), key=float, reverse=True)
    thresholds.append(min(thresholds) * 0.9)

    FPR = [];TPR = []
    for thre in thresholds:
        TP = 0;FN =0;FP = 0;TN = 0
        for value,proba in zip(y_true,y_proba):
            if proba>thre:
                if value ==1:
                    TP+=1
                else:
                    FP+=1
            else:
                if value == 1:
                    FN+=1
                else:
                    TN +=1

        FPR.append(FP/(FP+TN))
        TPR.append(TP/(TP+FN))

    return FPR,TPR,thresholds

y_test = [1,1,0,0,1]
y_proba = [0.5, 0.6, 0.55, 0.4, 0.7] # 对应分成分成1的概率
# """
# 计算ROC曲线
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
fpr, tpr, thresholds = roc_curve_custom(y_test, y_proba)
# 计算auc分数
roc_auc = auc(fpr, tpr)
print(roc_auc)
print(thresholds)
# 画roc曲线
plt.plot(fpr,tpr,c="r")
plt.show()
# """
```
