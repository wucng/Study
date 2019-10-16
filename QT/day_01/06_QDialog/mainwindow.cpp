#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    QMenuBar *mBar=menuBar();
    setMenuBar(mBar);
    QMenu *menu=mBar->addMenu("对话框");
    QAction *p1=menu->addAction("模态对话框");

    connect(p1,&QAction::triggered,[=](){
        QDialog dlg;
        dlg.exec(); // 窗口卡住 等待用户处理
    });

    QAction *p2=menu->addAction("非模态对话框");

    connect(p2,&QAction::triggered,[=](){
        QDialog *p=new QDialog;
        p->setAttribute(Qt::WA_DeleteOnClose);
        p->show();
    });


    QAction *p3=menu->addAction("关于对话框");
    connect(p3,&QAction::triggered,[=](){
        QMessageBox::about(this,"about","关于QT");
    });

    QAction *p4=menu->addAction("问题对话框");
    connect(p4,&QAction::triggered,[=](){
        int ret=QMessageBox::question(this,"question","Are you ok?");
        switch (ret) {
        case QMessageBox::Yes:
            cout<<"OK";
            break;
        case QMessageBox::No:
            cout<<"NO";
            break;
        default:
            break;
        }
    });

    // 自己指定对话框
    QAction *p5=menu->addAction("问题对话框2");
    connect(p5,&QAction::triggered,[=](){
        int ret=QMessageBox::question(this,"question","Are you ok?",
                                      QMessageBox::Ok|QMessageBox::Cancel);
        switch (ret) {
        case QMessageBox::Ok:
            cout<<"OK";
            break;
        case QMessageBox::Cancel:
            cout<<"Cancel";
            break;
        default:
            break;
        }
    });

    // 文件对话框
    QAction *p6=menu->addAction("文件对话框");
    connect(p6,&QAction::triggered,[=](){
        QString path=QFileDialog::getOpenFileName(
                    this,
                    "open",
                    "../",
                    "Images (*.png *.xpm *.jpg);;Text files (*.txt);;XML files (*.xml);;"
                    "souce (*.cpp *.h);;all (*.*)"
                    );
        cout<<path;
    });
}

MainWindow::~MainWindow()
{
}

