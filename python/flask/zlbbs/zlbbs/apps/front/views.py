from flask import Blueprint,views,render_template,make_response
from utils.captcha import Captcha
from io import BytesIO

bp = Blueprint("front",__name__)

@bp.route('/')
def index():
    return 'front index'

@bp.route("/captcha")
def graph_captcha(): # 图片保存在内存中在发布到网络(不是从磁盘中生成静态文件)
    # 获取验证码
    test,image = Captcha.gene_graph_captcha()
    # BytesIO 字节流
    out = BytesIO()
    image.save(out,'png') # 保存到BytesIO
    out.seek(0) # 从索引0开始读
    resp = make_response(out.read())
    resp.content_type = 'image/png'
    return resp


class SignupView(views.MethodView):
    def get(self):
        return render_template("front/front_signup.html")


bp.add_url_rule('/signup/',view_func=SignupView.as_view("signup"))