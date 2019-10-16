#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //设置窗口信息
    this->setWindowIcon(QIcon("../images/qt.ico"));
    this->setWindowTitle("图片标定系统");
    // this->setGeometry(200,200,500,500);
    this->resize(500,500);

    //安装事件过滤器
    this->installEventFilter(this);

    // 初始化矩形坐标
    x=0;
    y=0;
    w=0;
    h=0;

    // 菜单栏
    QMenuBar *mBar=menuBar();
    // 添加菜单
    QMenu *pFile=mBar->addMenu("文件");
    // 添加菜单项，添加动作
    QAction *pOpen=pFile->addAction("open");
    pOpen->setIcon(QIcon("../images/opened.ico"));


    // 状态栏
    QStatusBar *sBar= statusBar();
    // QLabel *label=new QLabel(this);
    label=new QLabel(this);
    label->setText("F1查看帮助");
    sBar->addWidget(label);


    // 点击open打开需标定的图片目录
    connect(pOpen,&QAction::triggered,[=]()->void{
        this->files=QFileDialog::getOpenFileNames(this,"open","../images/example",
                                      "Images (*.png *.xpm *.jpg *.jpeg);;"
                                      "All (*.*)");
        if(files.length()==0)
        {
            QMessageBox::warning(this,"打开失败","请确认选择合适的文件");
            return;
        }

        // 取第一张作为窗口背景，用于标定
        this->index=0;
        this->path=this->files.at(index);

        // 显示状态信息
        // label->setText("打开成功");
    });

    // 关闭窗口
    connect(this,&MainWindow::destroyed,[=](){
        // int ret=QMessageBox::question(this,"about","Are you sure?");
        // if(ret==QMessageBox::Yes)
        //    this->close();
        QMessageBox::about(this,"about","欢迎再次使用！");
    });

}

bool MainWindow::eventFilter(QObject *obj,QEvent *e)
{
    // 通过窗体的鼠标事件改变 x,y,w,h的值，实现动态画矩形
    if(obj==this)//主窗体
    {
        // QPaintEvent *env=static_cast<QPaintEvent *>(e);// 画图事件没法使用事件过滤器

        QMouseEvent *env=static_cast<QMouseEvent *>(e);
        //判断事件
        if(e->type()==QEvent::MouseButtonPress)//鼠标按下事件
        {
            // 左键画图(右键保存)
            if(env->button()==Qt::LeftButton) //左键
            {
                this->x=env->x();
                this->y=env->y();

                return true;
            }
            else
            {
                return QWidget::eventFilter(obj,e);
            }
        }
        else if(e->type()==QEvent::MouseButtonRelease)//鼠标释放
        {
            if(env->button()==Qt::LeftButton) //左键
            {
                this->w=env->x()-this->x;
                this->h=env->y()-this->y;

                if(this->w>0 & this->h>0)
                    update();//重绘 间接调用paintEvent()
                /*
                if(this->w>0 & this->h>0)
                {
                    update();//重绘 间接调用paintEvent()
                    return true;
                }
                else
                {
                    QMessageBox::warning(this,"error","请从左上方往右下方画图！");
                    return true;
                }
                */
                return true;

            }
            else
            {
                return QWidget::eventFilter(obj,e);
            }
        }
        else if(e->type()==QEvent::MouseButtonDblClick)//鼠标双击
        {
            if(env->button()==Qt::RightButton) //双击右键 保存
            {
                if(this->w>0 & this->h>0)
                {
                    // 保存
                    //关联文件
                    QString tmp=QString(this->path);
                    wFile.setFileName(tmp.append(".txt"));
                    //wFile=new QFile(this->path.append(".txt"));
                    bool isOK=wFile.open(QIODevice::WriteOnly);
                    if(isOK==true)
                    {
                        wFile.write((this->path+
                        QString(" x=%1,y=%2,w=%3,h=%4").arg(x).arg(y).arg(w).arg(h)).toLocal8Bit());

                        // cout<<"sucess";
                        // 显示状态信息
                        label->setText("保存成功");
                    }

                    wFile.close();
                    return true;
                }
                return QWidget::eventFilter(obj,e);
            }
            return QWidget::eventFilter(obj,e);
        }

        else if(e->type()==QEvent::KeyPress)// 键盘事件
        {   QKeyEvent *kenv=static_cast<QKeyEvent *>(e);
            if (kenv->key()==Qt::Key_Up)//↑ 上一张
            {
                this->index--;
                if(this->index<0)
                    this->index=0;
                this->path=this->files.at(this->index);

                update();
                return true;
            }
            else if (kenv->key()==Qt::Key_Down) // 下一张
            {

                this->index++;
                if(this->index>this->files.length()-1)
                    this->index=this->files.length()-1;
                this->path=this->files.at(this->index);

                update();
                return true;
            }
            else if (kenv->key()==Qt::Key_Q) // 退出
            {
                this->close();
                return true;
            }
            else if (kenv->key()==Qt::Key_F1) // F1 帮助信息
            {
                QMessageBox::about(this,"help",
                                   "1、文件->open 打开文件\n"
                                   "2、画图：按下鼠标从左上角拖动到右下角释放鼠标\n"
                                   "3、保存：双击鼠标右键\n"
                                   "4、按up键切换前一张,按down键切换下一张\n"
                                   "5、按q关闭\n"
                                   "6、按F1查看帮助\n"
                                   );
                return true;
            }
            else
                return QWidget::eventFilter(obj,e);
        }
        else
        {
            return QWidget::eventFilter(obj,e);
        }

    }
    return QWidget::eventFilter(obj,e);
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter p; //创建画家对象
    p.begin(this); //指定当前窗口为绘图设备

    // 画出背景
    p.drawPixmap(this->rect(),QPixmap(this->path));

    //定义画笔
    QPen pen;
    pen.setWidth(5);//设置线宽
    // pen.setColor(QColor(14,9,255));//rgb 设置颜色
    pen.setColor(Qt::red);//设置颜色
    pen.setStyle(Qt::DotLine);
    //把画笔交给画家
    p.setPen(pen);

    // 创建画刷对象
    QBrush brush;
    brush.setColor(Qt::gray);
    brush.setStyle(Qt::Dense4Pattern);
    // 把画刷交给画家
    p.setBrush(brush);

    // 画矩形
    p.drawRect(x,y,w,h);

    p.end();
}

MainWindow::~MainWindow()
{
    delete ui;
}

