#ifndef SMALLWDIGHT_H
#define SMALLWDIGHT_H

#include <QWidget>

namespace Ui {
class SmallWdight;
}

class SmallWdight : public QWidget
{
    Q_OBJECT

public:
    explicit SmallWdight(QWidget *parent = nullptr);
    ~SmallWdight();

    // 设置数字
    void setNum(int);
    // 获取数字
    int getNum();

private:
    Ui::SmallWdight *ui;
};

#endif // SMALLWDIGHT_H
