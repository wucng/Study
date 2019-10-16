#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    //菜单栏
    QMenuBar *mBar=menuBar();
    // QMenuBar *mBar=new QMenuBar(this);

    //添加菜单
    QMenu *pFile=mBar->addMenu("文件");

    //添加菜单项，添加动作
    QAction *pNew =pFile->addAction("新建");

    connect(pNew,&QAction::triggered,[](bool checked)->void{
        cout<<"pNew:"<<checked;
    });

    pFile->addSeparator(); // 添加分割线
    QAction *pOpen =pFile->addAction("打开");

    // 工具栏，菜单项的快捷键
    QToolBar *toolBar=addToolBar("toolBar");

    // 工具栏添加快捷键
    toolBar->addAction(pNew);
    toolBar->addAction(pOpen);

    QPushButton *b=new QPushButton(this);
    b->setText("^_^");

    // 添加控件
    toolBar->addWidget(b);

    connect(b,&QPushButton::pressed,[b](){
        b->setText("123");
    });

    // 状态栏
    QStatusBar *sBar= statusBar();
    QLabel *label=new QLabel(this);
    label->setText("Normal text file");
    sBar->addWidget(label);

    //addWidget 从左往右添加
    sBar->addWidget(new QLabel("2",this));

    //addPermanentWidget 默认从右往左添加
    sBar->addPermanentWidget(new QLabel("3",this));

    // 核心控件
    QTextEdit *textEdit = new QTextEdit(this);
    setCentralWidget(textEdit);

    //浮动窗口
    QDockWidget *dock =new QDockWidget(this);
    addDockWidget(Qt::LeftDockWidgetArea,dock);

    QTextEdit *textEdit1=new QTextEdit(this);
    dock->setWidget(textEdit1);

    // 如果浮动窗口被关闭了，点击打开会显示
    connect(pOpen,&QAction::triggered,[=](){
        dock->show();
    });

}

MainWindow::~MainWindow()
{
}

