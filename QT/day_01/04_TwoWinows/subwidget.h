#ifndef SUBWIDGET_H
#define SUBWIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QDebug>
#include <QString>
//#include <QTime>

#define cout qDebug()<<"["<<__FILE__<<":"<<__LINE__<<"]"

class subWidget : public QWidget
{
    Q_OBJECT
public:
    explicit subWidget(QWidget *parent = nullptr);

signals:
    // 信号重载
    void subWSignal();// 子窗体发送信号，让主窗体响应
    void subWSignal(QString,int);// 子窗体发送信号，让主窗体响应

public slots:

private:
    QPushButton b; // 单击按钮 切换窗口
};

#endif // SUBWIDGET_H
