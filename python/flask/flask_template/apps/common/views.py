# -*- coding:utf-8 -*-
# author:
# time:
# desc:路由，视图函数


from flask import Blueprint

bp = Blueprint("common",__name__,url_prefix='/common')

@bp.route('/')
def index():
    return 'common index'
