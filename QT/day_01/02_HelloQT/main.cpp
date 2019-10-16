#include <QApplication>
#include <QWidget> // 窗口控件基类
#include <QPushButton> // 按钮

int main(int argc,char **argv)
{
    QApplication app(argc,argv);

    QWidget w;
    w.setWindowTitle("Hello World"); // 设置标题
    w.show();

    QPushButton b;
    b.setText("button"); // 给按钮设置内容
    b.show();

    app.exec();
    return 0;
}
