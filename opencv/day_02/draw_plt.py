import matplotlib.pyplot as plt

train_loss =[0.623565,0.322954,0.239795,0.190909,0.159174,0.131932,0.110403,0.104337,0.088653,0.087797]
train_acc =[0.903843,0.947452,0.955301,0.961951,0.966258,0.971218,0.976124,0.976124,0.979395,0.979013]

test_loss =[0.1956, 0.0900, 0.0623,0.0478,0.0320, 0.0265,0.0323,0.0294,0.0315,0.0302]
test_acc =[4503/4585,4557/4585,4548/4585,4558/4585,4570/4585,4567/4585,4558/4585,4559/4585,4555/4585,4554/4585]

x = list(range(10))
plt.subplot(121)
l1,=plt.plot(x,train_loss)
l2,=plt.plot(x,test_loss)
plt.legend([l1, l2], ['train_loss', 'test_loss'], loc = 'upper right')

plt.subplot(122)
l1,=plt.plot(x,train_acc)
l2,=plt.plot(x,test_acc)
# plt.legend([l1, l2], ['train_acc', 'test_acc'], loc = 'upper right')
plt.legend([l1, l2], ['train_acc', 'test_acc'], loc = 'lower right')
plt.show()