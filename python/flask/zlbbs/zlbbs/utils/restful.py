from flask import jsonify

class HttpCode(object):
    ok = 200
    unautherror = 401
    paramserror = 400
    servererror = 500

def restful_result(code,message,data):
    return jsonify({"code":code,"message":message,"data":data or {}})

def success(message="",data=None):
    return restful_result(HttpCode.ok,message or "成功",data)

def unauthError(message="",data=None):
    return restful_result(HttpCode.unautherror,message or "没有授权",data)

def paramsError(message="",data=None):
    return restful_result(HttpCode.paramserror, message or "参数错误", data)

def serverError(message="",data=None):
    return restful_result(HttpCode.servererror, message or "服务器内部错误", data)