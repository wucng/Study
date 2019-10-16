#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    timerId=this->startTimer(1000);//毫秒为单位 每个1s触发定时器

    // button 按钮
    b.setParent(this);
    b.setIcon(QIcon("../cat.ico"));

    // 安装事件过滤器
    ui->label_2->installEventFilter(this);
    ui->label_2->setMouseTracking(true); // 获取焦点
}

void Widget::keyPressEvent(QKeyEvent *event)
{
    // cout<<(char)event->key();
    if(event->key()==Qt::Key_A) // 按下A键
    {
        cout<<"Qt::Key_A";
    }

}

void Widget::timerEvent(QTimerEvent *)
{
    static int sec=0;
    ui->label->setText(
                QString("<center><h1>timer out: %1</h1></center>").arg(sec++)
                );
    if(5==sec)
    {
        // 停止定时器
        this->killTimer(this->timerId);
    }

}

void Widget::mousePressEvent(QMouseEvent *event)
{
    // 获取鼠标按下的坐标
    position_x=event->x();
    position_y=event->y();

    // 根据坐标移动button按钮
    b.move(position_x,position_y);
}

void Widget::mouseMoveEvent(QMouseEvent *event)
{
    // 获取鼠标移动的坐标
    position_x=event->x();
    position_y=event->y();

    // 根据坐标移动button按钮
    b.move(position_x,position_y);
}

void Widget::closeEvent(QCloseEvent *event)
{
    int ret =QMessageBox::question(this,"question","是否需要关闭窗口？");
    if(ret==QMessageBox::Yes)
    {
        // 关闭窗口
        // 处理关闭窗口事件，接收事件，事件不会再往下传递
        event->accept();
    }
    else
    {
        //不关闭窗口
        // 忽略事件，事件继续传递给父组件
        event->ignore();
    }
}


bool Widget::event(QEvent *event)
{
    // 事件分发
    /*
    switch (event->type()) {
    case QEvent::Close:
        closeEvent(event);
        break;

    case QEvent::MouseMove:
        mouseMoveEvent(event);
        break;

    default:
        break;
    }
    */

    if(event->type()==QEvent::Timer)
    {
        //干掉定时器
        //如果返回true,事件停止传播
        QTimerEvent *env=static_cast<QTimerEvent *>(event);
        timerEvent(env);
        return true;
    }
    else if(event->type()==QEvent::KeyPress)
    {
        //类型转换
        QKeyEvent *env=static_cast<QKeyEvent *>(event);
        if(env->key()==Qt::Key_B)
        {
            return QWidget::event(env);
        }
        return true;
    }
    else
    {
        return QWidget::event(event);
    }
}


bool Widget::eventFilter(QObject *obj,QEvent *e)
{
    if(obj==ui->label_2)
    {
        QMouseEvent *env=static_cast<QMouseEvent *>(e);
        //判断事件
        if(e->type()==QEvent::MouseMove)
        {
            ui->label_2->setText(QString("Mouse Move:(%1,%2)").
                arg(env->x(),env->y()));
            return true; // true 阻止事件继续往下传(传递给父组件)
        }
        else{
            return QWidget::eventFilter(obj,e);
        }
    }
    else
    {
       return QWidget::eventFilter(obj,e);
    }
}

Widget::~Widget()
{
    delete ui;
}

