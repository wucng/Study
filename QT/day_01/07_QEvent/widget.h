#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QTimer>
#include <QMessageBox>
#include <QPushButton>
#include <QEvent>

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
    int timerId;

    // 点击窗口获取坐标，再根据这个坐标移动button按钮
    int position_x;
    int position_y;
    QPushButton b;

protected:
    //键盘按下事件
    void keyPressEvent(QKeyEvent *event) override;
    // 计时器事件
    void timerEvent(QTimerEvent *) override;

    void mousePressEvent(QMouseEvent *event) override;

    void mouseMoveEvent(QMouseEvent *event) override;

    void closeEvent(QCloseEvent *event) override;

    // 重写event事件（不推荐）
    bool event(QEvent *event) override;

    // 事件过滤器(推荐)
    bool eventFilter(QObject *,QEvent *) override;

};
#endif // WIDGET_H
