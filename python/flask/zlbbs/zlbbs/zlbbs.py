# config.py/exts.py/models.py/manage.py
# 前台、后台、公共的

# 有内置的开发好的网页，可以直接拿来使用
# https://www.bootcss.com/
# （Bootstrap3中文文档--->起步-->找到需要的页面-->右键 查看网页源代码-->再复制到自己的项目修改下即可）

from flask import Flask
from apps.cms import bp as cms_bp
from apps.common import bp as common_bp
from apps.front import bp as front_bp
import config
from exts import db,mail
from flask_wtf import CSRFProtect

def create_app():
    app = Flask(__name__)
    app.config.from_object(config) # 导入配置文件

    # 注册蓝图
    app.register_blueprint(cms_bp)
    app.register_blueprint(common_bp)
    app.register_blueprint(front_bp)

    db.init_app(app)
    mail.init_app(app)
    CSRFProtect(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
