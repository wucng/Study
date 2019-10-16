#include "mywidget.h"

MyWidget::MyWidget(QWidget *parent)
    : QWidget(parent)
{
    b1.setParent(this); // 设置父类
    b1.setText("close");
    b1.move(100,100);

    b2=new QPushButton(this); // 分配内存时指定父类
    b2->setText("===b2===");

    // 信号与槽 使用lambda表达式处理槽函数时 connect第3个参数(接收者) 可以省略
//    connect(&b1,&QPushButton::clicked,this,
//            [=](){
//        this->setWindowTitle("123");
//    });

    //
    connect(&b1,&QPushButton::clicked,
            [=](){
        // this->setWindowTitle("123");
        b2->setText("Hello");
    });

    this->resize(500,500);
}

MyWidget::~MyWidget()
{
    delete b2;
}

