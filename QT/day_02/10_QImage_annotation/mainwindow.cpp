#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 调用文件对话框选择图片
    // 点击open打开需标定的图片目录
    connect(ui->pOpen,&QAction::triggered,[=]()->void{
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


    // 设置鼠标最终状态
    setMouseTracking(true); //  默认进入聚焦

}


//窗体绘图事件
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

    // 画定位线
    p.drawLine(QPoint(location_x,0),QPoint(location_x,this->rect().height()));
    p.drawLine(QPoint(0,location_y),QPoint(this->rect().width(),location_y));

    // 画矩形
    // p.drawRect(x,y,w,h);
    p.drawRects(rects);

    p.end();
}


// 鼠标按下
void MainWindow::mousePressEvent(QMouseEvent *ev)
{
    if(ev->button() == Qt::LeftButton) //左键
    {
        this->x = ev->x();
        this->y = ev->y();
    }

}
// 鼠标释放
void MainWindow::mouseReleaseEvent(QMouseEvent *ev)
{
    if(ev->button() == Qt::LeftButton) //左键
    {
        this->w = ev->x()-this->x;
        this->h = ev->y()-this->y;

        // 记录每次的 x,y,w,h
        rects.append(QRect(x,y,w,h));

        // 手动调用 窗体绘图事件
        update();
    }

}

// 鼠标移动（显示定位线）
void MainWindow::mouseMoveEvent(QMouseEvent *ev)
{
    if(ev->buttons() & Qt::RightButton) //右键拖动
    {
        this->location_x = ev->x();
        this->location_y = ev->y();

        // qDebug()<<"鼠标移动: x= "<<location_x<<","<<"y = "<<location_y;

        // 手动调用 窗体绘图事件
        update();
    }

}

MainWindow::~MainWindow()
{
    delete ui;
}

