# -*- coding:utf-8 -*-
# author:
# time:
# desc:路由，视图函数

from flask import Blueprint

# bp = Blueprint("cms",__name__,subdomain="cms") # 子域名"cms"，需要修改host文件
bp = Blueprint("cms",__name__,url_prefix='/cms')

@bp.route('/')
def index():
    return "cms.index"
