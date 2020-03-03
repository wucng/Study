#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QWidget>
#include <QPushButton>
#include "mypushbutton.h"

class myWidget : public QWidget
{
    Q_OBJECT

public:
    myWidget(QWidget *parent = nullptr);
    ~myWidget();
};
#endif // MYWIDGET_H
