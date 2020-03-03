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
#include <QMouseEvent>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    //窗体绘图事件
    void paintEvent(QPaintEvent *);

    // 鼠标按下
    void mousePressEvent(QMouseEvent *ev);
    // 鼠标释放
    void mouseReleaseEvent(QMouseEvent *ev);
    // 鼠标移动
    void mouseMoveEvent(QMouseEvent *ev);

private:
    Ui::MainWindow *ui;

    QString path; // 记录当前文件路径
    QStringList files; // 保存打开文件夹下所有图片路径
    int index;//记录图片索引号

    QVector<QRect> rects;
    int x,y,w,h;//用于画矩形

    QFile wFile;// 保存文件
    int location_x,location_y; //定位线

};
#endif // MAINWINDOW_H
