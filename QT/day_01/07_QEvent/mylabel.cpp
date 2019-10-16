#include "mylabel.h"

MyLabel::MyLabel(QWidget *parent) : QLabel(parent)
{
    //设置追踪鼠标
    this->setMouseTracking(true);
    // this->setGeometry(0,0,300,300); // 左上角相对于窗体0,0，大小为300x300
    this->setPixmap(QPixmap("../test.jpg"));
}

void MyLabel::mousePressEvent(QMouseEvent *ev)
{
    // 相对于控件MyLabel的坐标
    int i=ev->x();
    int j=ev->y();

    //类似与sprintf使用
    QString str=QString("x:%1 y:%2").arg(i).arg(j);
    // cout<<str;

    if(ev->button()==Qt::LeftButton) //单击左键
    {
        cout<<"单击左键:"<<str;
    }
    else if(ev->button()==Qt::RightButton) //右键
    {
        cout<<"单击右键:"<<str;
    }
    else if(ev->button()==Qt::MidButton) // 鼠标中间键
    {
        cout<<"单击中间键:"<<str;
    }

    QString text=QString("<center><h1>Mouse Press: (%1,%2)</h1></center>").arg(i).arg(j);
    this->setText(text);
}


void MyLabel::mouseReleaseEvent(QMouseEvent *ev)
{
    QString text=QString("<center><h1>Mouse Release:(%1,%2)</h1></center>").arg(
                ev->x()).arg(ev->y());
    this->setText(text);
}

void MyLabel::mouseMoveEvent(QMouseEvent *ev)
{
    QString text=QString("<center><h1>Mouse Move:(%1,%2)</h1></center>").arg(
                ev->x()).arg(ev->y());
    this->setText(text);
}

void MyLabel::enterEvent(QEvent *)
{
    QString text=QString("<center><h1>Mouse Enter</h1></center>");
    this->setText(text);
}

void MyLabel::leaveEvent(QEvent *)
{
    QString text=QString("<center><h1>Mouse Leave</h1></center>");
    this->setText(text);
}
