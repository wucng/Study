
- [离散傅里叶变换-DFT（FFT基础）](https://blog.csdn.net/zhangxz259/article/details/81627341)
- [深入理解离散傅里叶变换(DFT)](https://zhuanlan.zhihu.com/p/71582795)
- [图像的二维DFT及其反变换](https://blog.csdn.net/carson2005/article/details/6586583)
---
# 一维DFT
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191220170012258.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)

  DFT的公式：
![](https://img-blog.csdn.net/20160303205423480?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191220170600275.png)
![](https://img-blog.csdn.net/20160303205523163?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

接下来就是对公式写程序了，先将公式展开：
![](https://img-blog.csdn.net/20160303210111697?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
在计算机中可以这样展开
![](https://img-blog.csdn.net/20160303210156541?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
Real就是cos相关的幅值，imag就是sin相关的幅值

最后将sin与cos合成一个sin，

![](https://img-blog.csdn.net/20160303210359288?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

# 二维DFT
欧拉公式：
$e^{ix} = cos(x)+isin(x)$


$F(u,v)=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi (ux/M+vy/N))}$



$e^{-j2\pi (ux/M+vy/N))}=cos(2\pi(ux/M+vy/N))-jsin(2\pi(ux/M+vy/N))$


u值（u=0,1,2,...,M-1）和v值(v=0,1,2,..,N-1)计算

$F(u,v)=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)cos(2\pi(ux/M+vy/N))-j\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)sin(2\pi(ux/M+vy/N))$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191220174633363.png)

---
==傅里叶反变换==

$f(x,y)=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}F(u,v)e^{j2\pi (ux/M+vy/N))}$


x=0,1,...,M-1,y=0,1,...,N-1;

$f(x,y)=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}F(u,v)cos(2\pi(ux/M+vy/N))+j\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}F(u,v)sin(2\pi(ux/M+vy/N))$

$F(u,v)$ 分为实数部分(real)，虚数部分（image）

$F(u,v)=real(u,v)+j*image(u,v)$

$f(x,y)=\frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}[real(u,v)cos(2\pi(ux/M+vy/N))-image(u,v)sin(2\pi(ux/M+vy/N))] + \frac{1j}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}[image(u,v)cos(2\pi(ux/M+vy/N))+real(u,v)sin(2\pi(ux/M+vy/N)))]$

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019122017460433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3djNzgxNzA4MjQ5,size_16,color_FFFFFF,t_70)

----