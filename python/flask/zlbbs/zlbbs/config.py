#encoding: utf-8
import os
from datetime import timedelta

# PERMANENT_SESSION_LIFETIME = timedelta(hours=2) # 手动设置cookie保存时间

SECRET_KEY = os.urandom(24)

DEBUG = True

HOSTNAME = "127.0.0.1"
PORT = "3306"
DATABASE = "zlbbs"
USERNAME = "root"
PASSWORD = "123456"

# dialect+driver://username:password@host:port/database
DB_URI = "mysql+pymysql://{username}:{password}@{host}:{port}/{db}?charset=utf8".format(username=USERNAME,password=PASSWORD,host=HOSTNAME,port=PORT,db=DATABASE)

SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False


CMS_USER_ID = 'ASDFASDFSA'


# MAIL_USE_TLS：端口号587
# MAIL_USE_SSL：端口号465
# QQ邮箱不支持非加密方式发送邮件
# 发送者邮箱的服务器地址
MAIL_SERVER = "smtp.qq.com"
MAIL_PORT = '587'
MAIL_USE_TLS = True
# MAIL_USE_SSL
MAIL_USERNAME = "2413357360@qq.com"
MAIL_PASSWORD = "ghsnqpxaneujdjdg"
MAIL_DEFAULT_SENDER = "2413357360@qq.com"
