# -*- coding:utf-8 -*-
# author:
# time:
# desc:数据库模型


from exts import db
from datetime import datetime
from werkzeug.security import generate_password_hash,check_password_hash
import shortuuid

class CMSUser(db.Model):
    __tablename__ = 'cms_user'
    # id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id = db.Column(db.String(100), primary_key=True, default=shortuuid.uuid)
    username = db.Column(db.String(50), nullable=False)
    _password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(50), nullable=False, unique=True)
    telephone = db.Column(db.String(11), nullable=False, unique=True)
    premission = db.Column(db.Integer,default=0)
    join_time = db.Column(db.DateTime, default=datetime.now)

    __mapper_args__ = {
        "order_by": join_time.desc() # 降序
    }

    def __init__(self, username, password, email,telephone):
        self.username = username
        self.password = password
        self.email = email
        self.telephone = telephone

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, raw_password):
        self._password = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        result = check_password_hash(self.password, raw_password)
        return result