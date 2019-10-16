#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QPainter>
#include <QBitmap>
#include <QPen>
#include <QBrush>
#include <QEvent>
#include <QMouseEvent>
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

private:
    Ui::Widget *ui;

    int x;

    QPoint a[5]; //画五边形用

protected:
    //重写绘图事件，虚函数
    // 如果在窗口绘图，必须放在绘图事件里实现
    //绘图事件内部自动调用，窗口需要重绘的时候（状态改变）
    void paintEvent(QPaintEvent *event);

    // 事件过滤器(推荐) 实现单击窗体 画不规则图形
    bool eventFilter(QObject *,QEvent *);

private slots:
    void on_pushButton_pressed();
};
#endif // WIDGET_H
