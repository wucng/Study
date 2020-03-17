# -*- coding:utf-8 -*-
# author:
# time:
# desc:表单验证

from wtforms import StringField,IntegerField,ValidationError
from wtforms.validators import Email,InputRequired,Length,EqualTo,Regexp
from wtforms import Form
# from flask import g

# 登录
class LoginForm(Form):
    telephone = StringField(validators=[Regexp(r"1[345789]\d{9}",message='请输入正确格式的手机号码！')])
    password = StringField(validators=[Regexp(r"[0-9a-zA-Z_\.]{6,20}",message='请输入正确格式的密码！')])
    remember = IntegerField()

    def get_error(self):
        message = self.errors.popitem()[1][0]
        return message

# 注册
class LogupForm(Form):
    telephone = StringField(validators=[Regexp(r"1[345789]\d{9}",message='请输入正确格式的手机号码！')])
    email = StringField(validators=[Email(message="请输入正确的邮箱"),InputRequired(message="请输入邮箱")])
    username = StringField(validators=[Regexp(r".{2,20}", message='请输入正确格式的用户名！')])
    password = StringField(validators=[Regexp(r"[0-9a-zA-Z_\.]{6,20}",message='请输入正确格式的密码！')])
    password2 = StringField(validators=[EqualTo("password",message='两次输入的密码不一致！')])

    def get_error(self):
        message = self.errors.popitem()[1][0]
        return message