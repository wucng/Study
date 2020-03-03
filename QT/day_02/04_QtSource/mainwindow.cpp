#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

//    ui->actionnew->setIcon(QIcon("../xxx.jpg"));

    //使用添加Qt资源 （在线Icon图片下载 https://www.easyicon.net/）
    // 1.将图片拷到项目下
    //2.右击项目 --> 添加新文件 --> Qt -->Qt Resource File
    // 右键xxx.qrc --> Open in Editor -->添加 -->添加前缀 -->添加 -->添加文件

    //使用添加Qt资源 ": + 前缀名 + 文件名"
    ui->actionnew->setIcon(QIcon(":/images/new.ico"));
    ui->actionopen->setIcon(QIcon(":/images/open.ico"));

}

MainWindow::~MainWindow()
{
    delete ui;
}

