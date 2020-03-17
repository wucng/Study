# -*- coding:utf-8 -*-
# author:
# time:
# desc:管理数据库
"""
$ mysql -uroot -p
# 创建数据库
mysql> create database zlbbs charset utf8;

# 终端运行manage.py 实现数据库迁移
# 初始化
$ python manage.py db init

# 生成一个迁移脚本
$ python manage.py db migrate

# 更新（映射）到数据库中
$ python manage.py db upgrade
"""


from flask_script import Manager
from flask_migrate import Migrate,MigrateCommand
from app import create_app
from exts import db
from apps.cms import models as cms_models

CMSUser = cms_models.CMSUser

app = create_app()

manager = Manager(app)

Migrate(app,db)
manager.add_command('db',MigrateCommand)


# 终端创建用户 $ python manage.py create_cms_user -u zhiliao -p 111111 -e zhiliao@qq.com -t 18888888888
@manager.option('-u','--username',dest='username')
@manager.option('-p','--password',dest='password')
@manager.option('-e','--email',dest='email')
@manager.option('-t','--telephone',dest='telephone')
def create_cms_user(username,password,email,telephone):
    user = CMSUser(username=username,password=password,email=email,telephone=telephone)
    db.session.add(user)
    db.session.commit()
    print('cms用户添加成功！')


# 修改用户权限
# @manager.command


if __name__ == '__main__':
    manager.run()