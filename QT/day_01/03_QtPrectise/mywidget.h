#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QWidget>
#include <QPushButton>

class MyWidget : public QWidget
{
    Q_OBJECT

public:
    MyWidget(QWidget *parent = nullptr);
    ~MyWidget();

public slots:
    void mySlot(); //自定义槽函数

signals:
    void mySignal();//自定义信号


private:
    // QPushButton *b;
    QPushButton b2;
};
#endif // MYWIDGET_H
