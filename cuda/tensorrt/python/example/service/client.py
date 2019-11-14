import requests
import base64
# import cv2


def img2base64(imgPath):
    with open(imgPath, 'rb') as f:
        image = base64.b64encode(f.read())

    # return image
    return bytes.decode(image)

imgPath="/media/wucong/work/practice/data/test/5.JPG"

url="http://192.168.199.113:7001/run"
json={"img64":img2base64(imgPath),"path":imgPath}
r=requests.post(url=url,json=json)
print(r.json())