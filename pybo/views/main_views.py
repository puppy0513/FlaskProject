from flask import Blueprint, render_template, url_for , request
from werkzeug.utils import redirect


bp = Blueprint('main', __name__, url_prefix='/')

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
        f.save(secure_filename(f.filename))
        return 'uploads 디렉토리 -> 파일 업로드 성공!'




