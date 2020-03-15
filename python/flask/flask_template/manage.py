# -*- coding:utf-8 -*-
# author:
# time:
# desc:数据库迁移文件

from flask_script import Manager
from flask_migrate import Migrate,MigrateCommand
from zlbbs import create_app
from exts import db
from apps.cms import models as cms_models
from apps.front import models as front_models

CMSUser = cms_models.CMSUser
CMSRole = cms_models.CMSRole
CMSPermission = cms_models.CMSPermission

FrontUser = front_models.FrontUser

app = create_app()

manager = Manager(app)
Migrate(app,db)
manager.add_command('db',MigrateCommand)

# 通过终端命令添加用户
@manager.option('-u',"--username",dest='username')
@manager.option('-p',"--password",dest='password')
@manager.option('-e',"--email",dest='email')
def create_cms_user(username,password,email):
    user = CMSUser(username=username,password=password,email=email)
    db.session.add(user)
    db.session.commit()
    print("cms用户添加成功！")


@manager.command
def create_role(): # python manage.py create_role
    # 1.访问者(可以修改个人信息)
    vistor = CMSRole(name='访问者',desc='只能访问数据，不能修改')
    vistor.permissions = CMSPermission.VISITOR

    # 运营角色(可以修改个人信息，可以管理帖子和评论，前台用户)
    operator = CMSRole(name="运营",desc="管理帖子,评论和前台用户")
    operator.permissions = CMSPermission.VISITOR|CMSPermission.POSTER|\
                           CMSPermission.CMSUSER|CMSPermission.COMMENTER|CMSPermission.FORNTUSER

    # 3、管理员(拥有绝大部分权限)
    admin = CMSRole(name="管理员",desc="拥有本系统所有权限")
    admin.permissions = CMSPermission.VISITOR|CMSPermission.POSTER|\
                           CMSPermission.CMSUSER|CMSPermission.COMMENTER|\
                        CMSPermission.FORNTUSER|CMSPermission.BOARDER

    # 4.开发者
    developer = CMSRole(name="开发者",desc="开发人员专用角色")
    developer.permissions = CMSPermission.ALL_PERMISSION

    db.session.add_all([vistor,operator,admin,developer])
    db.session.commit()


@manager.option('-e',"--email",dest='email')
@manager.option('-n',"--name",dest='name')
def add_user_to_role(email,name): # python manage.py add_user_to_role -e zhiliao@qq.com -n 访问者/开发者
    user = CMSUser.query.filter_by(email=email).first()
    if user:
        role = CMSRole.query.filter_by(name=name).first()
        if role:
            role.users.append(user)
            db.session.commit()
            print("用户添加到角色成功！")
        else:
            print("没有这个角色：%s"%role)
    else:
        print("%s邮箱没有这个用户！"% email)

@manager.option('-t','--telephone',dest='telephone')
@manager.option('-u','--username',dest='username')
@manager.option('-p','--password',dest='password')
def create_front_user(telephone,username,password):
    user = FrontUser(telephone=telephone,username=username,password=password)
    db.session.add(user)
    db.session.commit()



@manager.command
def test_permission():
    user = CMSUser.query.first()
    if user.has_permissions(CMSPermission.VISITOR):
        print("这个用户有访问者的权限！")
    else:
        print("这个用户没有访问者的权限！")
if __name__=="__main__":
    manager.run()