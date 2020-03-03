#ifndef TEARCHER_H
#define TEARCHER_H

#include <QObject>

class tearcher : public QObject
{
    Q_OBJECT
public:
    explicit tearcher(QObject *parent = nullptr);

signals:
    //自定义信号 写到signals下
    // 返回值是void，只需要声明，不需要实现
    // 可以有参数，可以重载
    void hungry();

    void hungry(QString foodName);

public slots:
};

#endif // TEARCHER_H
