#ifndef STUDENT_H
#define STUDENT_H

#include <QObject>

class student : public QObject
{
    Q_OBJECT
public:
    explicit student(QObject *parent = nullptr);

signals:

public slots:
    // 早期Qt版本 必须写到public slots 下
    // 高级版本可以写到 public或全局下
    // 返回值 void,需要声明和实现
    //可以有参数，可以重载
    void treat();

    void treat(QString foodName);
};

#endif // STUDENT_H
