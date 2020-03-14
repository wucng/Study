from wtforms import StringField,IntegerField,ValidationError
from wtforms.validators import Email,InputRequired,Length,EqualTo
from ..forms import BaseForm
from utils import zlcache
from flask import g

class LoginForm(BaseForm):
    email = StringField(validators=[Email(message="请输入正确的邮箱"),
                                    InputRequired(message="请输入邮箱")])
    password = StringField(validators=[Length(6,20,message="密码必须是6~20位")])
    remember = IntegerField()

class ResetpwdForm(BaseForm):
    oldpwd = StringField(validators=[Length(6,20,message="密码必须是6~20位")])
    newpwd = StringField(validators=[Length(6,20,message="密码必须是6~20位")])
    newpwd2 = StringField(validators=[EqualTo("oldpwd",message="新密码不一致")])


class ResetEmailForm(BaseForm):
    email = StringField(validators=[Email(message='请输入正确格式的邮箱！')])
    captcha = StringField(validators=[Length(min=6,max=6,message='请输入正确长度的验证码！')])

    def validate_captcha(self,field):
        captcha = field.data
        email = self.email.data
        captcha_cache = zlcache.get(email)
        if not captcha_cache or captcha.lower() != captcha_cache.lower():
            raise ValidationError('邮箱验证码错误！')

    def validate_email(self,field):
        email = field.data
        user = g.cms_user
        if user.email == email:
            raise ValidationError('不能修改为相同的邮箱！')















