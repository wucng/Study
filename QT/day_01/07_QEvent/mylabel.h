#ifndef MYLABEL_H
#define MYLABEL_H

#include <QLabel>
#include <QMouseEvent>
#include <QEvent>
#include <QDebug>
#define cout qDebug()<<"["<<__FILE__<<":"<<__LINE__<<"]"

class MyLabel : public QLabel
{
    Q_OBJECT
public:
    explicit MyLabel(QWidget *parent = nullptr);

signals:

public slots:

// 重写事件
protected:
    // 鼠标点击事件
    void mousePressEvent(QMouseEvent *ev) override;
    // 鼠标移动事件
    void mouseMoveEvent(QMouseEvent *ev) override;
    // 鼠标释放事件
    void mouseReleaseEvent(QMouseEvent *ev) override;
    // 进入窗口区域
    void enterEvent(QEvent *ev) override;
    // 离开窗口区域
    void leaveEvent(QEvent *ev) override;
};

#endif // MYLABEL_H
