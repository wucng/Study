#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QDebug>
#include "subwidget.h"

#define cout qDebug()<<"["<<__FILE__<<":"<<__LINE__<<"]"

class mainWidget : public QWidget
{
    Q_OBJECT

public:
    mainWidget(QWidget *parent = nullptr);
    ~mainWidget();

private:
    subWidget subW;//声明子窗口
    QPushButton b; // 单击按钮 切换窗口
};
#endif // MAINWIDGET_H
