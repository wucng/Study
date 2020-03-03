#include "smallwdight.h"
#include "ui_smallwdight.h"

SmallWdight::SmallWdight(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SmallWdight)
{
    ui->setupUi(this);

    ui->horizontalSlider->setMaximum(100);
    ui->horizontalSlider->setMinimum(0);
    //QSpinBox移动 QSlider跟着移动
    void(QSpinBox::*spSignal )(int) = &QSpinBox::valueChanged;
    connect(ui->spinBox,spSignal,ui->horizontalSlider,&QSlider::setValue);

    // QSlider滑动 QSpinBox跟着移动
    connect(ui->horizontalSlider,&QSlider::valueChanged,ui->spinBox,&QSpinBox::setValue);


}

SmallWdight::~SmallWdight()
{
    delete ui;
}

// 设置数字
void SmallWdight::setNum(int num)
{
    ui->spinBox->setValue(num);
}
// 获取数字
int SmallWdight::getNum()
{
    return ui->spinBox->value();
}
