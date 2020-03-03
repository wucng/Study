#include "widget.h"
#include "ui_widget.h"
#include <QTimer>
#include <QPushButton>
#include <QMouseEvent>
#include <QDebug>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    //启动定时器
    id1 = startTimer(1000);//单位 毫秒

    id2 = startTimer(2000);//单位 毫秒

    //定时器第二种方式
    QTimer *timer = new QTimer(this);
    // 启动定时器
    timer->start(500);//单位 毫秒

    connect(timer,&QTimer::timeout,[=](){
        static int num =1;
        //label4 每隔0.5秒 +1
        ui->label_4->setText(QString::number(num++));
    });

    // 点击暂停按钮 实现停止定时器
    connect(ui->btn,&QPushButton::clicked,[=](){
        if(ui->btn->text()=="暂停")
        {
            timer->stop();
            ui->btn->setText("启动");
        }
        else {
            timer->start(500);
            ui->btn->setText("暂停");
        }
    });


    // 给label1安装事件过滤器
    //步骤1 安装事件过滤器
    ui->label->installEventFilter(this);
    // 步骤2 重写 eventFiler
}
// 重写事件过滤器的事件
bool Widget::eventFiler(QObject *obj,QEvent *e)
{
    if(obj==ui->label)
    {
        if(e->type()==QEvent::MouseButtonPress)
        {
            QMouseEvent *ev = static_cast<QMouseEvent *>(e);
            QString str = QString("鼠标按下 x = %1 y = %2 ").arg(ev->x()).arg(ev->y());
            qDebug()<<str.toUtf8().data();

            return true;//true 代表用户自己处理事件，不向下分发
        }
    }

    //其他的事件 交给父类处理 默认处理
    return QWidget::eventFilter(obj,e);
}


Widget::~Widget()
{
    delete ui;
}

void Widget::timerEvent(QTimerEvent *ev)
{
    if(ev->timerId()==id1)
    {
        static int num = 1;
        // label2每隔1秒+1
        ui->label_2->setText(QString::number(num++));
    }

    if(ev->timerId()==id2)
    {
        // label3每隔2秒+1
        static int num2 = 1;
        ui->label_3->setText(QString::number(num2++));
    }

}
