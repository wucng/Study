from flask import (Blueprint,views,render_template,
                   request,session,redirect,url_for,g,
                   jsonify)
from .forms import LoginForm,ResetpwdForm,ResetEmailForm
from .models import CMSUser,CMSPermission
from .decorators import login_required,permission_required
import config
from exts import db,mail
from flask_mail import Message
from utils import restful,zlcache
import string
import random

# bp = Blueprint("cms",__name__,subdomain="cms") # 子域名"cms"，需要修改host文件
bp = Blueprint("cms",__name__,url_prefix='/cms')

# @bp.route('/',endpoint="index")
@bp.route('/')
@login_required
def index():
    return render_template("cms/cms_index.html")

@bp.route('/logout/')
@login_required
def logout():
    # session.clear()
    del session[config.CMS_USER_ID]
    return redirect(url_for("cms.login"))


@bp.route('/profile/')
@login_required
def profile():
    return render_template("cms/cms_profile.html")


@bp.route("/email_captcha/")
def email_captcha():
    # /email_captcha/?email=xxxx@qq.com
    email = request.args.get("email")
    if not email:
        return restful.paramsError("请传递邮箱参数！")

    source = list(string.ascii_letters)
    source.extend(list(map(str,[0,1,2,3,4,5,6,7,8,9])))
    captcha = "".join(random.sample(source,6))

    # 给邮箱发送邮件
    message = Message('python论坛邮箱验证码',recipients=[email],body="您的验证码是：%s"%(captcha))
    try:
        mail.send(message)
    except:
        return restful.serverError()

    # 存到memcached
    zlcache.set(email,captcha)
    # print(zlcache.get(email))
    return restful.success()

@bp.route('/email/')
@login_required
def send_email():
    message = Message("邮件发送",recipients=["781708249@qq.com"],body="测试")
    mail.send(message)

    return restful.success()


@bp.route('/posts/')
@login_required
@permission_required(CMSPermission.POSTER)
def posts():
    return render_template("cms/cms_posts.html")

@bp.route('/comments/')
@login_required
@permission_required(CMSPermission.COMMENTER)
def comments():
    return render_template("cms/cms_comments.html")

@bp.route('/boards/')
@login_required
@permission_required(CMSPermission.BOARDER)
def boards():
    return render_template("cms/cms_boards.html")

@bp.route('/fusers/')
@login_required
@permission_required(CMSPermission.FORNTUSER)
def fusers():
    return render_template("cms/cms_fusers.html")

@bp.route('/cusers/')
@login_required
@permission_required(CMSPermission.CMSUSER)
def cusers():
    return render_template("cms/cms_cusers.html")

@bp.route('/croles/')
@login_required
@permission_required(CMSPermission.ALL_PERMISSION)
def croles():
    return render_template("cms/cms_croles.html")

class LoginView(views.MethodView):
    def get(self,message=None):
        return render_template('cms/cms_login.html',message=message)

    def post(self):
        form = LoginForm(request.form)
        if form.validate():
            email = form.email.data
            password = form.password.data
            remember = form.remember.data
            user = CMSUser.query.filter_by(email=email).first()
            if user and user.check_password(password):
                session[config.CMS_USER_ID] = user.id
                if remember:
                    # 如果session.permanent = True
                    # 那么cookie过期时间为31天
                    session.permanent = True
                return redirect(url_for('cms.index'))
            else:
                return self.get(message="邮箱或密码错误")

        else:
            # print(form.errors)
            # message = form.errors.popitem()[1][0]
            message = form.get_error()
            return self.get(message=message)

class ResetPwdView(views.MethodView):
    decorators = [login_required] # 验证登录
    def get(self):
        return render_template("cms/cms_resetpwd.html")

    def post(self):
        form = ResetpwdForm(request.form)
        if form.validate():
            newpwd = form.newpwd.data
            oldpwd = form.oldpwd.data
            user = g.cms_user
            if user.check_password(oldpwd):
                user.password = newpwd
                db.session.commit()
                # {"code":200,"message":"密码错误"}
                # return jsonify({"code":200,"message":""})
                return restful.success()
            else:
                # return jsonify({"code": 400, "message": "旧密码验证错误"})
                return restful.paramsError("旧密码验证错误")
        else:
            # message = form.errors.popitem()[1][0]
            # message = form.get_error()
            # return jsonify({"code":400,"message":message})
            return restful.paramsError(form.get_error())


class ResetEmailView(views.MethodView):
    decorators = [login_required]  # 验证登录
    def get(self):
        return render_template("cms/cms_resetemail.html")
    def post(self):
        form = ResetEmailForm(request.form)
        if form.validate():
            email = form.email.data
            g.cms_user.email = email
            db.session.commit()
            return restful.success()
        else:
            return restful.paramsError(form.get_error())

bp.add_url_rule("/login/",view_func=LoginView.as_view('login'))
bp.add_url_rule("/resetpwd/",view_func=ResetPwdView.as_view('resetpwd'))
bp.add_url_rule("/resetemail/",view_func=ResetEmailView.as_view('resetemail'))

