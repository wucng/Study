#include "widget.h"
#include "ui_widget.h"

// 字符编码指针
QTextCodec *codec=nullptr;

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    this->setGeometry(500,500,500,500);

    // 初始化
    codec=QTextCodec::codecForName("GBK");
}

Widget::~Widget()
{
    delete ui;
}


void Widget::on_buttonRead_clicked()
{
    QString path=QFileDialog::getOpenFileName(this,"open","../",
                                              "Images (*.png *.xpm *.jpg);;"
                                              "Text files (*.txt);;"
                                              "XML files (*.xml);;"
                                               "souce (*.cpp *.h);;all (*.*)"
                                              );
    if(path.isEmpty()==false)
    {
        //文件对象
        QFile file(path);

        //打开文件，只读方式
        bool isOK=file.open(QIODevice::ReadOnly);
        if(isOK==true)
        {
            #if 0
            //读文件，默认只识别utf8编码
            QByteArray array=file.readAll();
            //显示到编辑区
            ui->textEdit->setText(codec->toUnicode(array));//转成utf8编码
            // ui->textEdit->setText(QString(array));
            // codec->fromUnicode(array); //utf-8-->GBK
            #endif

            QByteArray array;
            while (file.atEnd()==false) {
                //读一行
                array+=file.readLine();
            }
            ui->textEdit->setText(codec->toUnicode(array));//转成utf8编码

        }

        //关闭文件
        file.close();
    }
}



void Widget::on_buttonWrite_clicked()
{
    QString path =QFileDialog::getSaveFileName(this,"save","../",
                                               "Images (*.png *.xpm *.jpg);;"
                                               "Text files (*.txt);;"
                                               "XML files (*.xml);;"
                                                "souce (*.cpp *.h);;all (*.*)"
                                               );
    if(path.isEmpty()==false)
    {
        QFile file;//创建文件对象
        //关联文件
        file.setFileName(path);

        //打开文件，只写方式
        bool isOK=file.open(QIODevice::WriteOnly);
        if(isOK==true)
        {
            //获取编辑区内容
            QString str=ui->textEdit->toPlainText();

            /*
            //写文件
            // QString ->QByteArray
            file.write(str.toUtf8());

            // QString --> C++ string -->char *
            file.write(str.toStdString().data());
            */

            // 转换为本地平台编码
            file.write(str.toLocal8Bit());

            /*
            // QString -->QByteArray
            QString buf="123";
            QByteArray a=buf.toUtf8();//中文
            a=buf.toLocal8Bit();// 本地编码

            // QByteArray-->char *
            char *b=a.data();

            // char* -->QString
            char *p="abc";
            QString c=QString(p);
            */

        }

        file.close();
    }
}

void Widget::on_pushButton_clicked()
{
    on_buttonWrite_clicked();
}
