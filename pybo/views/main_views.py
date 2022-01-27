from flask import Blueprint, render_template, url_for , request
from werkzeug.utils import redirect
from werkzeug.utils import secure_filename
import os
from flask import Flask, request


bp = Blueprint('main', __name__, url_prefix='/')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from pybo.models import Question

@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo!'

@bp.route('/')
def index():
    return redirect(url_for('question._list'))

# 업로드 HTML 렌더링
@bp.route('/upload')
def render_file():
    return render_template('upload.html')

# 파일 업로드 처리
@bp.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # 저장할 경로 + 파일명
        f.save("./pybo/uploads/" + secure_filename(f.filename))
        return render_template('upload2.html')


