#include "mywidget.h"

myWidget::myWidget(QWidget *parent)
    : QWidget(parent)
{
    // 使用自定义的按钮控件创建按钮控件
    myPushButton *btn = new myPushButton(this);

    // 显示文本
    btn->setText("第一个按钮");

    resize(600,400);

}

myWidget::~myWidget()
{
}

