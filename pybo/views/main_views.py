from flask import Blueprint, render_template, url_for
from werkzeug.utils import redirect


bp = Blueprint('main', __name__, url_prefix='/')

from pybo.models import Question

@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo!'

@bp.route('/')
def index():
    return redirect(url_for('question._list'))




