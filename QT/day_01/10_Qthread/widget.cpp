#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    /*
    myTimer=new QTimer(this);

    //只要定时器启动，自动触发timeout()
    connect(myTimer,&QTimer::timeout,[=](){
        static int i=0;
        i++;
        //设置lcd的值
        ui->lcdNumber->display(i);
    });
    */

    //动态分配空间，不能指定父对象
    myT=new MyThread;

    //创建子线程
    thread=new QThread(this);

    //把自定义的线程加入到子线程中
    myT->moveToThread(thread);

    connect(myT,&MyThread::mySignal,[=](){
        static int i=0;
        i++;
        ui->lcdNumber->display(i);
    });

    cout<<"主线程号："<<QThread::currentThread();

//    connect(this,&Widget::startThread,[=](){
//        myT->myTimeout();
//    }); // 这里直接使用lambda表达式会报错

    connect(this,&Widget::startThread,myT,&MyThread::myTimeout);


    connect(this,&Widget::destroyed,[=](){
        on_buttonStop_clicked();
        delete myT;
    });
}

Widget::~Widget()
{
    delete ui;
}

/*
void Widget::on_pushButton_clicked()
{
    //如果定时器没有工作
    if(myTimer->isActive()==false)
    {
        myTimer->start(100);
    }

    //非常复杂的数据处理，耗时较长
    QThread::sleep(5);

    //处理完成后，关闭定时器
    myTimer->stop();
    cout<<"over";
}
*/



void Widget::on_buttonStart_clicked()
{
    if(thread->isRunning()==true)
    {
        return;
    }

    //启动线程，但是没有启动线程处理函数
    thread->start();
    myT->setFlag(false);

    //不能直接调用线程处理函数
    //直接调用 导致 线程处理函数和主线程在同一个线程
    // myT->myTimeout();

    //只能通过 signal-slot方法调用
    emit startThread();
}

void Widget::on_buttonStop_clicked()
{
    if (thread->isRunning()==false)
        return;

    myT->setFlag(true);
    thread->quit();
    thread->wait();
}
