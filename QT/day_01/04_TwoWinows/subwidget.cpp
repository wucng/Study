#include "subwidget.h"

subWidget::subWidget(QWidget *parent) : QWidget(parent)
{
    this->resize(400,200);
    this->setWindowTitle("子窗口");
    this->setWindowIcon(QIcon("../cat.ico"));

    b.setParent(this);//button 设置父对象
    b.setText("切换到主窗口");

    // 子窗体发送信号，主窗体接收信号，实现子窗体切换到主窗体
    connect(&b,&QPushButton::pressed,[this]()->void{
        cout<<"子窗体给主窗体发送信号";
        emit this->subWSignal();
        emit this->subWSignal("子窗口",20);
    });
}
