# -*- coding:utf-8 -*-
# author:
# time:
# desc:视图函数

from flask import Blueprint,render_template,views,request,session,redirect,url_for,g
from utils import restful
from exts import db
from .decorators import login_required
import config
from .forms import LoginForm,LogupForm
from .models import CMSUser

# bp = Blueprint("cms",__name__,subdomain="cms") # 子域名"cms"，需要修改host文件
bp = Blueprint("cms",__name__,url_prefix='/cms')

@bp.route('/')
# @login_required  # 强制必须登录
def index():

    return render_template("cms/index.html")

@bp.route('/details/')
# @login_required  # 强制必须登录
def details():
    users = db.session.query(CMSUser).all() # 倒序排序
    return render_template("cms/details.html",users=users)

# @bp.route('/delete/<telephone>')
@bp.route('/delete/',methods=["GET","POST"])
def delete():
    telephone = request.form.get("telephone")
    user = db.session.query(CMSUser).filter(CMSUser.telephone==telephone).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        # return redirect(url_for("cms.index"))
        return restful.success()
    else:
        return restful.paramsError(message="用户不存在！无法删除！")


"""
@bp.route("/login/")
def login():
    return render_template("cms/login.html")

@bp.route("/logup/")
def logup():
    return render_template("cms/logup.html")
"""

class LoginView(views.MethodView):
    # decorators = [login_required]  # 验证登录
    def get(self):
        return render_template("cms/login.html")
    def post(self):
        # telephone = request.form.get("telephone")
        # password = request.form.get("password")
        # remember = request.form.get("remember")
        form = LoginForm(request.form)
        if form.validate():
            telephone = form.telephone.data
            password = form.password.data
            remember = form.remember.data
            user = CMSUser.query.filter_by(telephone=telephone).first()
            if user and user.check_password(password):
                session[config.CMS_USER_ID] = user.id

                if remember:
                    # 如果session.permanent = True
                    # 那么cookie过期时间为31天
                    session.permanent = True
                    # return redirect(url_for('cms.index')) # ajax做页面切换，这里不做
                    return restful.success()
            else:
                return restful.paramsError(message="手机号或密码错误")
        else:
            message = form.get_error()
            return restful.serverError(message=message)


class LogupView(views.MethodView):
    def get(self):
        return render_template("cms/logup.html")
    def post(self):
        form = LogupForm(request.form)
        if form.validate():
            telephone = form.telephone.data
            email = form.email.data
            username = form.username.data
            password = form.password.data
            user = CMSUser.query.filter_by(telephone=telephone).first()
            if user: # 存在该用户
                return restful.paramsError(message="该手机号已经注册了，不能重复注册！")
            else: # 不存在则添加到数据库
                user = CMSUser(username,password,email,telephone)
                db.session.add(user)
                db.session.commit()
                return restful.success()

        else:
            message = form.get_error()
            return restful.serverError(message=message)




bp.add_url_rule("/login/",view_func=LoginView.as_view('login'))
bp.add_url_rule("/logup/",view_func=LogupView.as_view('logup'))




@bp.before_request
def before_request(): # 会首先执行该函数
    if config.CMS_USER_ID in session:
        user_id = session.get(config.CMS_USER_ID)
        user = CMSUser.query.get(user_id)
        if user:
            g.cms_user = user

'''
@bp.context_processor
def cms_context_processor(): # 将后台的东西传递到网页，不必使用传参函数
    return {"CMSPermission":CMSPermission}
'''