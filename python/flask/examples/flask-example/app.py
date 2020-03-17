from flask import Flask
from apps.cms import bp as cms_bp
from flask_wtf import CSRFProtect
from exts import db
import config

# app = Flask(__name__)
# app.config.from_object(config)
# csrf = CSRFProtect(app)

def create_app():
    app = Flask(__name__)
    app.config.from_object(config) # 导入配置文件

    # 注册蓝图
    app.register_blueprint(cms_bp)

    db.init_app(app)
    # mail.init_app(app)
    CSRFProtect(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()