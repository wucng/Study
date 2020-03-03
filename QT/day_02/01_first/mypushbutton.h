#ifndef MYPUSHBUTTON_H
#define MYPUSHBUTTON_H

#include <QWidget>
#include <QPushButton>

class myPushButton : public QPushButton
{
    Q_OBJECT
public:
    explicit myPushButton(QWidget *parent = nullptr);

signals:

public slots:
};

#endif // MYPUSHBUTTON_H
