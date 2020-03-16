"""
# flask error CSRF token is missing
# https://stackoverflow.com/questions/48673308/flask-error-csrf-token-is-missing

app = Flask(__name__)
csrf = CSRFProtect(app)

@app.route('/some-view', methods=['POST'])
@csrf.exempt
def some_view():
    ...

bp = Blueprint(...)
csrf.exempt(bp)
"""

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
    csrf = CSRFProtect(app)

    # 注册蓝图
    csrf.exempt(cms_bp)
    csrf.exempt(common_bp)
    csrf.exempt(front_bp)
    app.register_blueprint(cms_bp)
    app.register_blueprint(common_bp)
    app.register_blueprint(front_bp)

    db.init_app(app)
    mail.init_app(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
