#include "widget.h"
#include "ui_widget.h"
#include <QPainter> //画家类
#include <QPushButton>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    // 点击移动按钮 移动图片
    connect(ui->pushButton,&QPushButton::clicked,[=](){
        posX+=20;
        //如果要手动调用绘图事件 用update();
        update();
    });
}

Widget::~Widget()
{
    delete ui;
}

//绘图事件
void Widget::paintEvent(QPaintEvent *event)
{
    //实例化画家对象 this指定的是绘图设备
    //利用画家 画资源图片
    QPainter painter(this);

    // 如果超出屏幕 从0 开始
    if(posX>this->width())
    {
        posX = 0;
    }

    painter.drawPixmap(posX,0,QPixmap(":/images/open.ico"));
}
