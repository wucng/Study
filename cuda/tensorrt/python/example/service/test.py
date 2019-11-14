import json
from bottle import route, run, template, request, get, post, static_file,response
import bottle

@post("/")
@route("/")
def doRun():
    return json.dumps({"name":"123","score":"0.976"})

if __name__=="__main__":
    run(host='0.0.0.0', port=7001, debug=False, server='paste',reloader=True)