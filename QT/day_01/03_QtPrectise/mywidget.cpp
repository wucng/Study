#include "mywidget.h"
#include <QPushButton>
#include <QDebug>

#define cout qDebug()<<"["<<__FILE__<<":"<<__LINE__<<"]"

MyWidget::MyWidget(QWidget *parent)
    : QWidget(parent)
{

    // 设置窗口大小
    this->resize(500,500);
    this->setWindowTitle("sinal and slot");// 设置标题
    this->setWindowIcon(QIcon("../Bear.ico")); // 设置图标

    QPushButton *b=new QPushButton(this); // 指定父对象
    //b=new QPushButton(this);
    b->setText("窗口最大化");
    b->move(50,50); // x,y

    // or
    // QPushButton b2;
    b2.setParent(this); // 指定父对象
    b2.setText("窗口正常");
    b2.move(250,50);

    // 信号与槽使用
    connect(b,&QPushButton::clicked,this,&MyWidget::showFullScreen);

    connect(&b2,&QPushButton::clicked,this,&MyWidget::showNormal);

    // 使用lambda表达式，槽函数可以不用接收者
    connect(this,&MyWidget::destroyed, //窗口关闭触发
            [](){
        cout<<"close success!";
    }
            );

    QPushButton *b3=new QPushButton(this);
    b3->setText("关闭窗口");
    b3->move(50,250); // x,y
    // 自定义信号与槽函数（自定义信号实现窗口关闭）
    // 单击button 窗体发出一个自定义的信号，然后窗体接收信号，调用相应的槽函数处理（实现button移动）
    connect(b3,&QPushButton::pressed,this,&MyWidget::mySlot);

    connect(this,&MyWidget::mySignal,this,&MyWidget::close);

    // 直接关闭
    QPushButton *b4=new QPushButton(this);
    b4->setText("关闭窗口");
    b4->move(250,250); // x,y
    connect(b4,&QPushButton::pressed,this,&MyWidget::close);
    /*
     * b4 信号发送者
     * QPushButton::pressed b4发送的信号
     * this(当前窗体MyWidget) 信号接收者
     * MyWidget::close 槽函数，接收者触发的动作
     * 信号与槽函数相关联起来，
     * 信号的返回值，参数类型 必须与 槽函数的返回值，参数类型 否则没法关联
     */
}

void MyWidget::mySlot()
{
    cout<<"发送信号";
    emit mySignal(); // 发送信号
}

MyWidget::~MyWidget()
{
}

