"""
数据下载：https://tianchi.aliyun.com/forum/postDetail?postId=3770
https://github.com/YouChouNoBB/data-mining-introduction/tree/master/sklearn
"""
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interp

# 加载数据
df = pd.read_csv("trainCG.csv").fillna(0) # 默认 sep=","
# df=pd.read_csv(path,sep=' ') sep="," ； txt文件 sep=' '

X = df.drop("label",axis=1).astype(np.float64)
y = df.label # df["label"]

# 特征归一化
X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)


# 分类
cls = LogisticRegression().fit(X_train,y_train)
y_pred = cls.predict(X_test)

# 预测精度
acc = metrics.accuracy_score(y_test,y_pred)

#
scores = cls.predict_proba(X_test) # [m,2]
# scores = np.max(scores,-1)
metrics.roc_auc_score(y_test,scores[:,1]) # 取是1分数

# -----------------------------------------------------
# 画 auc图
# plot ROC curve and area the curve
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
cv = StratifiedKFold(n_splits=3)
classifier = LogisticRegression()
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1


# 画图
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()