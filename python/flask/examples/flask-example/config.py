import os
from datetime import timedelta

# PERMANENT_SESSION_LIFETIME = timedelta(hours=2) # 手动设置cookie保存时间

SECRET_KEY = os.urandom(24)

DEBUG = True

# 设置数据库连接
DB_USERNAME = 'root'
DB_PASSWORD = '123456'
DB_HOST = '127.0.0.1'
DB_PORT = '3306'
DB_NAME = 'zlbbs'

# PERMANENT_SESSION_LIFETIME =

DB_URI = 'mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8' % (DB_USERNAME,DB_PASSWORD,DB_HOST,DB_PORT,DB_NAME)

SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False


CMS_USER_ID = 'ASDFASDFSA'