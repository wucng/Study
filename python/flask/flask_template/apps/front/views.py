# -*- coding:utf-8 -*-
# author:
# time:
# desc:路由，视图函数

from flask import Blueprint,views,render_template,make_response
# from utils.captcha import Captcha
from io import BytesIO

bp = Blueprint("front",__name__)

@bp.route('/')
def index():
    return 'front index'