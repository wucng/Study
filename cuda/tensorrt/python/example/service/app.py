import json
from bottle import route, run, template, request, get, post, static_file,response
import bottle
import time
import os
import sys
import base64
import cv2
import numpy as np
import PIL.Image
# reload(sys)
# sys.setdefaultencoding('utf8')
from data_prepare import test_transformations
from model import MyModel,category
import torch
import torch.nn.functional as F

# 加上这个修饰器，解决跨域访问
def allow_cross_domain(fn):
    def _enable_cors(*args, **kwargs):
        #set cross headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,OPTIONS'
        allow_headers = 'Referer, Accept, Origin, User-Agent'
        response.headers['Access-Control-Allow-Headers'] = allow_headers
        if bottle.request.method != 'OPTIONS':
            # actual request; reply with the actual response
            return fn(*args, **kwargs)
    return _enable_cors

def imagestring2cvimg(image_string):
    np_str = np.fromstring(image_string, np.uint8)
    return cv2.imdecode(np_str,cv2.IMREAD_COLOR ) # cv2.IMREAD_GRAYSCALE

@route('/')
def do():
    return template('index.html')

@post("/run")
@post("/run/")
@allow_cross_domain
def doRun():
    # start = time.time()
    try:
        response.content_type = 'application/json'
        # jdata = request.body.read()
        # data = json.loads(bytes.decode(jdata), encoding="utf-8")
        # data=eval(data)
        # imgdata = base64.b64decode(data["img64"])

        imgdata = base64.b64decode(request.params['img64'])
        img = imagestring2cvimg(imgdata)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR-->RGB
        img = PIL.Image.fromarray(img).convert("RGB")

        # path = data["path"]
        # img = PIL.Image.open(path).convert("RGB")
        img = test_transformations(img).unsqueeze(dim=0).to(device)

        with torch.no_grad():
            output=F.softmax(model(img),-1)
            # score, pred = output.max(1, keepdim=False) # Tpo1
            score, pred = output.topk(5, 1, True, True) # top5

        jdata = {}
        for s,p in zip(score[0],pred[0]):
            jdata[category[p.item()]]= "%.3f"%(s.item()) #str(s.item())
        # jdata["name"]=category[pred.item()]
        # jdata["score"]="%.3f"%(score.item())

    except Exception as e:
        print(e)
        return json.dumps({"success":0})
    # return json.dumps({"success":1}.update(jdata))
    return json.dumps({"success":1,**jdata})

if __name__=='__main__':
    num_classes = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MyModel(num_classes, "resnet18", 512, False, 0.0)
    model.to(device)
    state_dict = torch.load("./model.pt")
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
    model.eval()

    # pip3 install paste
    # run(host='0.0.0.0', port=7001, debug=False, server='paste')
    run(host='0.0.0.0', port=7001, debug=False, server='paste',reloader=True)