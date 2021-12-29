from flask import Flask, render_template, request, flash, redirect, url_for
from flask import render_template
from flask_bootstrap import Bootstrap
from MTCNN_FaceNet.easy_use import *
from logging import FileHandler, WARNING
import os, base64
from flask_cors import CORS
from werkzeug.utils import secure_filename
import secrets
from simplified.classify_test import *
from simplified.face_append import *
from datetime import timedelta

secret = secrets.token_urlsafe(32)


app = Flask(__name__, template_folder='./templates')
app.send_file_max_age_default = timedelta(seconds=1)
CORS(app, resources=r'/*')
app.secret_key = secret
bootstrap = Bootstrap(app)
app.config['DEBUG'] = True
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

def tojpg(image):
    if len(image.split()) == 4:
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    return image


@app.route('/', methods=['GET', 'POST'])
def upinfo():
    if request.method == 'POST':
        if request.files.get('photo'):
            # 创建文件夹，保存录入图片
            photo = request.files.get('photo')
            basepath = os.path.dirname(__file__)
            filename = secure_filename(photo.filename)
            uploadpath = os.path.join(basepath, 'static/DataBase', filename[:-4], filename)
            path = os.path.join(basepath, 'static/DataBase', filename[:-4])
            if not path:
                os.makedirs(path)

            Reshape = transforms.Resize((160, 160))
            trans = transforms.Compose([Reshape])
            img = trans(tojpg(Image.open(photo)))
            save_path = uploadpath
            newphoto = mtcnn_single(img, save_path=save_path)

            # 更新dataset.json
            args = {'image_path': uploadpath, "dataset_path": 'static/face_dataset.json', 'name': filename[:-4]}
            face_append(args)
            return render_template('aaa.html', output='DataBase/' + filename[:-4] + '/' + filename)

        if request.files['image']:
            photo = request.files['image']
            basepath = os.path.dirname(__file__)
            filename = secure_filename(photo.filename)
            uploadpath = os.path.join(basepath, 'static/screenshot', filename)
            photo.save(uploadpath + '.jpg')

            Reshape = transforms.Resize((160, 160))
            trans = transforms.Compose([Reshape])
            img = trans(tojpg(Image.open(photo)))
            save_path = 'static/recognized_screenshot/' + "recognized_" + filename + '.jpg'
            newphoto = mtcnn_single(img, save_path=save_path)

            uploadpath = os.path.join(basepath, 'static/recognized_screenshot', 'recognized_'+filename)
            args = {'img_path': uploadpath + '.jpg', 'dataset_path': 'static/face_dataset.json',
                    'origin_data': 'static/DataBase'}
            out = classify_test(args)
            if out != "no matched people":
                print("数据库存储路径：" + out)
                print("识别成功！")
            else:
                print(out)
                print("数据库中不存在该人脸信息！")

            return {'path': out}

    return render_template('aaa.html')


if __name__ == "__main__":
    app.run()
