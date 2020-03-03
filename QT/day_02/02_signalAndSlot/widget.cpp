#include "widget.h"
#include "ui_widget.h"
#include <QPushButton>
#include <QDebug>

//

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    // 创建一个老师对象
    this->zt = new tearcher(this);

    // 创建一个学生对象
    this->st = new student(this);

    /*
    // 老师饿了，学生请客的链接
    connect(zt,&tearcher::hungry,st,&student::treat);

    // 调用下课函数
    classIsOver();
    */

    /*

    // 函数指针 -> 函数地址
    void(tearcher::*tearcherSignal)(QString) = &tearcher::hungry;
    void(student::*studentSlot)(QString) = &student::treat;
    connect(zt,tearcherSignal,st,studentSlot);
    classIsOver();

    // 点击一个 下课的按钮，在触发下课
    QPushButton *btn = new QPushButton("下课",this);
    //重置窗口大小
    this->resize(600,400);

    //点击按钮 触发下课
    // connect(btn,&QPushButton::clicked,this,&Widget::classIsOver);

    // 无参信号和槽链接
    void(tearcher::*tearcherSignal2)(void) = &tearcher::hungry;
    void(student::*studentSlot2)(void) = &student::treat;
    connect(zt,tearcherSignal2,st,studentSlot2);

    // 信号连接信号
    connect(btn,&QPushButton::clicked,zt,tearcherSignal2);

    // 断开信号
    // disconnect(zt,tearcherSignal2,st,studentSlot2);

    //拓展
    //1.信号是可以连接信号
    //2.一个信号可以连接多个槽函数
    //3.多个信号 可以连接同一个槽函数
    //4.信号和槽函数的参数 必须类型一一对应
    //5.信号和槽的参数个数 信号的参数个数 可以多于槽函数的参数个数

    // Qt4 版本以前的信号和槽连接方式
    //利用Qt4信号槽 连接无参版本
    connect(zt,SIGNAL(hungry()),st,SLOT(treat()));
    //QT4版本优点，参数直观，缺点：类型不做检测
    // Qt5以上 支持Qt4的版本写法 反之不支持

    */

    // lambda表达式
    /*
    [=](){
        btn->setText("下课了");
    }(); // 加上()表示调用
    */

    /*
    // mutable 关键字 用于修饰值传递的变量，修改的是拷贝 而不是本身
    QPushButton *btn = new QPushButton("one",this);
    QPushButton *btn2 = new QPushButton("two",this);
    btn2->move(100,100);
    int m = 10;

    connect(btn,&QPushButton::clicked,this,
            [m]() mutable {
        m += 100;
        qDebug()<<m;
    });

    connect(btn2,&QPushButton::clicked,this,
            [=]() {
        qDebug()<<m;
    });

    qDebug()<<m;
    */

    // ->int 表示返回值类型（也可以省略）
    // int ret = []()->int{return 100;}();
    // qDebug()<<"ret = "<<ret;


    void(tearcher::*tearcherSignal)(QString) = &tearcher::hungry;
    void(student::*studentSlot)(QString) = &student::treat;
    connect(zt,tearcherSignal,st,studentSlot);

    //利用lambda表达式 实现点击按钮 关闭窗口
    QPushButton *btn = new QPushButton();
    btn->setText("关闭");
    btn->setParent(this);

//    connect(btn,&QPushButton::clicked,this,[=](){
//       this->close();//关闭窗口
//       emit zt->hungry("宫保鸡丁");
//    });

    connect(btn,&QPushButton::clicked,[=](){ //使用lambda表达式时，第3个参数是this可以省略
       this->close();//关闭窗口
       emit zt->hungry("宫保鸡丁");
    });

}

void Widget::classIsOver()
{
    // 下课函数，调用后 触发老师饿了的信号
    // emit zt->hungry();
    emit zt->hungry("宫保鸡丁");
}

Widget::~Widget()
{
    delete ui;
}

