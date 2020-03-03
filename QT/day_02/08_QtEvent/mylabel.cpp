#include "mylabel.h"
#include <QDebug>
#include <QMouseEvent>

myLabel::myLabel(QWidget *parent) : QLabel(parent)
{
    // 设置鼠标最终状态
    setMouseTracking(true);
}

//鼠标进入事件
void myLabel::enterEvent(QEvent *event)
{
    qDebug()<<"鼠标进入";
}
// 鼠标离开事件
void myLabel::leaveEvent(QEvent *event)
{
    qDebug()<<"鼠标离开";
}

// 鼠标按下
void myLabel::mousePressEvent(QMouseEvent *ev)
{
    //但鼠标左键按下 提示信息
    if(ev->button()==Qt::LeftButton)
    {
        QString str = QString("鼠标按下 x = %1 y = %2 ").arg(ev->x()).arg(ev->y());
        qDebug()<<str.toUtf8().data();
    }

}
// 鼠标释放
void myLabel::mouseReleaseEvent(QMouseEvent *ev)
{
    qDebug()<<"鼠标释放";
}
// 鼠标移动
void myLabel::mouseMoveEvent(QMouseEvent *ev)
{
//    if(ev->buttons() & Qt::LeftButton)
    {
        QString str = QString("鼠标移动 x = %1 y = %2 ").arg(ev->globalX()).arg(ev->globalY());
        qDebug()<<str.toUtf8().data();
    }
}

bool myLabel::event(QEvent *e)
{
    //如果是鼠标按下 在event事件分发中做拦截操作
    if(e->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent *ev = static_cast<QMouseEvent *>(e);
        QString str = QString("鼠标按下 x = %1 y = %2 ").arg(ev->x()).arg(ev->y());
        qDebug()<<str.toUtf8().data();

        return true;//true 代表用户自己处理事件，不向下分发
    }

    //其他的事件 交给父类处理 默认处理
    return QLabel::event(e);
}
