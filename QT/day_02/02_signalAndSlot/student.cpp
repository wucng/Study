#include "student.h"
#include <QDebug>

student::student(QObject *parent) : QObject(parent)
{

}

void student::treat()
{
    qDebug()<<"请老师吃饭";
}

void student::treat(QString foodNmae)
{
    //QString -> char *  先转成QByteArray( .toUTF8() ) 再转char *( .data() )
    qDebug()<<"请老师吃饭,老师要吃 "<<foodNmae.toUtf8().data();
}
