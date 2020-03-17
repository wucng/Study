# 数据库增删改成系统

- 1、用户注册
id telephone 邮箱 username password premission(权限)

- 2、用户登录
用户名/手机/邮箱  密码

- 3、显示用户信息


```python
pip install flask

pip install pymysql
pip install SQLAlchemy

# or
pip install flask-sqlalchemy
sudo apt-get install libmysqlclient-dev
pip3 install mysqlclient
pip3 install flask-mysqldb


pip install redis
pip install mysql
pip install flask-wtf # pip install wtforms
pip install flask-restful
pip install flask-script
pip install flask-migrate # pip install alembic
pip install flask-mail

# https://www.runoob.com/memcached/window-install-memcached.html
# https://memcached.org/downloads
pip install python-memcached

pip install shortuuid

pip install celery
pip install flask-paginate
pip install qiniu
```
```python
# 有内置的开发好的网页，可以直接拿来使用
# https://www.bootcss.com/
# （Bootstrap3中文文档--->起步-->找到需要的页面-->右键 查看网页源代码-->再复制到自己的项目修改下即可使用）
#
# 内置网页素材
# https://www.17sucai.com/

# ueditor文本编辑器
# http://fex.baidu.com/ueditor/
```
---
```
$ mysql -uroot -p
# 创建数据库
mysql> create database zlbbs charset utf8;


# 终端运行manage.py 实现数据库迁移
$ cd xxx/xxx/zlbbs
# 初始化
$ python manage.py db init

# 生成一个迁移脚本
$ python manage.py db migrate

# 更新（映射）到数据库中
$ python manage.py db upgrade

# 切换到数据库
mysql> use zlbbs;
mysql> show tables;

# 使用命令手动添加cms用户
$ python manage.py create_cms_user -u zhiliao -p 111111 -e 123@qq.com

# 查看用户
mysql> select * from cms_user;


# 更新数据库字段
$ python manage.py db migrate
$ python manage.py db upgrade
```

