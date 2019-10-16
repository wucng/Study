#include "mainwidget.h"

mainWidget::mainWidget(QWidget *parent)
    : QWidget(parent)
{
    this->resize(400,200);
    this->setWindowTitle("主窗口");
    this->setWindowIcon(QIcon("../Bear.ico"));

    b.setParent(this);//button 设置父对象
    b.setText("切换到子窗口");

    // subW.show(); // 显示子窗口

    //单击按钮切换窗口
    connect(&b,&QPushButton::pressed,
            [=]()->void{
        this->hide();
        subW.show();
    });

    // 主窗体接收子窗体的信号，切换窗口
    // 涉及到信号重载，需使用函数指针 指明具体是哪个信号
    void (subWidget::*funNoParam)()=&subWidget::subWSignal;
    void (subWidget::*funTwoParam)(QString,int)=&subWidget::subWSignal;

//    connect(&subW,&subWidget::subWSignal,[=]()->void{
//        cout<<"接收来自子窗口的信号";
//        this->show();
//        subW.hide();
//    });

    connect(&subW,funTwoParam,[=](QString str,int a)->void{
        cout<<"接收来自子窗口的信号:"<<str<<" "<<a;
        this->show();
        subW.hide();
    });
}

mainWidget::~mainWidget()
{
}

