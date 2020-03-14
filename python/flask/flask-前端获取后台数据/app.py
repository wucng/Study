from flask import Flask,g,session,render_template,jsonify,redirect,url_for,request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html",data={"username":"zhiliao"})

@app.route('/success/',methods=["GET","POST"]) # ,endpoint="succ", 没有设置endpoint，则默认endpoint为函数名
def success():
    return jsonify({"code":200,"message":"success","data":"123456"})
    # return redirect(url_for("hello_world"))

@app.before_request
def before_request():# 启动app会首先执行该钩子函数
    g.username = "zhiliaoketang"
    session.username = "zhiliaoketang"

@app.context_processor
def cms_context_processor(): # 将后台的东西传递到网页，不必使用传参函数
    return {"cont":{"code":200,"message":"成功","data":"123456"}}



if __name__ == '__main__':
    app.run()
