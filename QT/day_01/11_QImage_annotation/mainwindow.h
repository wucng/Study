#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QToolBar>
#include <QStatusBar>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QDockWidget>
#include <QFileDialog>
#include <QMessageBox>
#include <QEvent>
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QPaintEvent>
#include <QFile>

#define cout qDebug()<<"["<<__FILE__<<":"<<__LINE__<<"]"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QString path; // 记录当前文件路径
    QStringList files; // 保存打开文件夹下所有图片路径
    int index;//记录图片索引号

    int x,y,w,h;//用于画矩形

    QFile wFile;// 保存文件

    QLabel *label;//显示状态信息
protected:
    // 事件过滤器(推荐)
    bool eventFilter(QObject *,QEvent *);

    //重写绘图事件，虚函数
    // 如果在窗口绘图，必须放在绘图事件里实现
    //绘图事件内部自动调用，窗口需要重绘的时候（状态改变）
    void paintEvent(QPaintEvent *event);

};
#endif // MAINWINDOW_H
