#include "mythread.h"

MyThread::MyThread(QObject *parent) : QObject(parent)
{
    isStop=false;
}

void MyThread::myTimeout()
{
    while (isStop==false) {

        QThread::sleep(1);

        emit mySignal(); //任务处理完成后，向主线程发送信号

        cout<<"子线程号："<<QThread::currentThread();

        if(isStop==true)
            break;
    }
}

void MyThread::setFlag(bool flag)
{
    isStop=flag;
}
