#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QTimer> // 定时器头文件
#include <QThread>
#include "mythread.h"
#include <QDebug>
#define cout qDebug()<<"["<<__FILE__<<":"<<__LINE__<<"]"

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    // void dealTimeout(); //定时器槽函数

private slots:
//    void on_pushButton_clicked();

//    void on_pushButton_2_clicked();

    void on_buttonStart_clicked();

    void on_buttonStop_clicked();

signals:
    void startThread();//启动子线程的信号

private:
    Ui::Widget *ui;

    // QTimer *myTimer;//声明变量

    MyThread *myT;
    QThread *thread;
};
#endif // WIDGET_H
