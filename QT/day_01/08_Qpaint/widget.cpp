#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    x=0;

    // 如果想单击窗体改变笑脸位置，
    // 使用信号与槽函数,但是窗体没有鼠标点击信号，因此没法实现
    // 只能使用窗体的鼠标点击事件来做 (或者使用事件过滤器来处理)
    // connect(this,QWidget::)


    // 去窗口边框
    setWindowFlags(Qt::FramelessWindowHint | windowFlags());

    // 把窗口背景设置为透明
    setAttribute(Qt::WA_TranslucentBackground);

    //安装事件过滤器
    this->installEventFilter(this);

    // 设初始值
    for(int i=0;i<5;i++)
    {
        this->a[i]=QPoint(qrand()%300,qrand()%300);
    }

}


bool Widget::eventFilter(QObject *obj,QEvent *e)
{
    if(obj==this)
    {
        QMouseEvent *env=static_cast<QMouseEvent *>(e);
        //判断事件
        if(e->type()==QEvent::MouseButtonPress)
        {

            if (env->button()==Qt::RightButton)
            {
                // 如果是右键
                this->close();
                return true; // 事件不继续传递
            }
            else if(env->button()==Qt::LeftButton)
            {
                cout<<QString("(%1,%2)").arg(env->x()).arg(env->y());

                //鼠标左键点击窗口改变随机值
                for(int i=0;i<5;i++)
                {
                    this->a[i]=QPoint(qrand()%300,qrand()%300);
                }

                update();//重绘 间接调用paintEvent()

                return true;
                // return QWidget::eventFilter(obj,e);
            }
            else
            {
                return QWidget::eventFilter(obj,e);
            }


        }
        else
        {
            return QWidget::eventFilter(obj,e);
        }
    }
    else
    {
        return QWidget::eventFilter(obj,e);
    }
}

Widget::~Widget()
{
    delete ui;
}

void Widget::paintEvent(QPaintEvent *event)
{
    // QPainter p(this);
    // or
    QPainter p; //创建画家对象
    p.begin(this); //指定当前窗口为绘图设备

    //绘图操作

    // 画背景图
    // p.drawPixmap(0,0,this->width(),this->height(),QPixmap("../test.jpg"));
    // The rect property equals QRect(0, 0, width(), height())
    // p.drawPixmap(this->rect(),QPixmap("../test.jpg")); // QPixmap RGB
    // p.drawPixmap(this->rect(),QBitmap("../test.jpg"));// QBitmap 二值图

    QPixmap pixmap;
    pixmap.load("../test.jpg");
    p.drawPixmap(this->rect(),pixmap); // QPixmap RGB


    //定义画笔
    QPen pen;
    pen.setWidth(5);//设置线宽
    // pen.setColor(Qt::red);//设置颜色
    pen.setColor(QColor(14,9,255));//rgb 设置颜色
    pen.setStyle(Qt::DotLine);

    //把画笔交给画家
    p.setPen(pen);

    //画直线
    p.drawLine(50,50,150,50);
    p.drawLine(150,50,150,150);
    p.drawLine(150,150,50,150);
    p.drawLine(50,50,50,150);

    // 创建画刷对象
    QBrush brush;
    brush.setColor(Qt::gray);
    brush.setStyle(Qt::Dense4Pattern);

    // 把画刷交给画家
    p.setBrush(brush);

    // 画矩形
    p.drawRect(200,200,200,200);

    // 画圆形
    p.drawEllipse(300,300,100,100);

    // 画笑脸
    p.drawPixmap(x,500,80,80,QPixmap("../Bear.ico"));

    // 绘制五边形
    p.drawPolygon(this->a,5);

    p.end();
}

void Widget::on_pushButton_pressed()
{
    this->x+=20;
    if(x>this->width())
        x=0;

    // 刷新窗口，让窗口重绘
    update();// 间接调用paintEvent()
}
