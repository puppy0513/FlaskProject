import time
from flask import Blueprint, render_template, url_for , request, session, jsonify, g
from werkzeug.utils import redirect
from werkzeug.utils import secure_filename
import os
from flask import Flask, request
import pandas as pd
import json
from werkzeug.routing import BaseConverter, ValidationError
from synthpop import Synthpop
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


bp = Blueprint('main', __name__, url_prefix='/')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.debug = True

from pybo.models import Question
# from datasets.adult import df, dtypes
# from pybo.synthpop.datasets.adult import df, dtypes


@bp.route('/hello')
def hello_pybo():
    fff = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + g.user.username + ".csv")
    return fff.to_html()



@bp.route('/')
def index():
    return redirect(url_for('question._list'))

# 업로드 HTML 렌더링
@bp.route('/upload')
def render_file():
    return render_template('upload.html')

global obj

# 파일 업로드 처리
@bp.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # f = pd.DataFrame(data = request.files['file'])
        # 저장할 경로 + 파일명

        obj = g.user.username
        #  ff = pd.DataFrame(data = f)
        f.save("C:/finalproject/myproject/pybo/uploads/" + obj + '.csv')
        # fff = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + '.csv', encoding='CP949')
        fff = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + '.csv')
        # return f.to_html()
        df_info = fff
        df_col = []
        for i in range(0, len(df_info.columns)):
            df_col.append(df_info.columns[i])
    
        
        df2 = df_info.iloc[:10]

        return render_template('upload2.html', df_list=df2, df_col = df_col, df_info=df_info)


# json 생성
@bp.route('/to_json', methods=['GET', 'POST'])
def to_json():
        obj = g.user.username
        df_info = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
        df_col = []

        for i in range(0, len(df_info.columns)):
            df_col.append(df_info.columns[i])
        df_type = []
        for i in range(0, len(df_info.columns)):
            if (df_info.dtypes[i] == 'int64'):
                df_type.append('int')
            elif (df_info.dtypes[i] == 'float64'):
                df_type.append('float')
            else:
                df_type.append('category')

        to_json = {df_col[0]: df_type[0]}
        # 이렇게 dict로 주지 않으면 list형식으로 들어감 ;
        for i in range(1, len(df_info.columns)):
            to_json[df_col[i]] = df_type[i]
        with open("C:/finalproject/myproject/pybo/uploads/" + obj + ".json", 'w') as f:
            json.dump(to_json, f)
        
        return render_template('to_json.html', df=df_info, df_col=df_col)
    
# json 생성 및 재현데이터 생성
@bp.route('/synth_generate', methods=['GET', 'POST'])
def synth_generate():
        obj = g.user.username

        # df = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
        with open("C:/finalproject/myproject/pybo/uploads/" + obj + ".json", 'r') as f:
            dtypes = json.load(f)
        columns = list(dtypes.keys())

        df = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv", header=None, skiprows = 1, names=columns).astype(dtypes)
        # 헤더가 있는 경우 -> skip
        df.apply(pd.to_numeric, errors='coerce')

        spop = Synthpop()
        spop.fit(df, dtypes)

        synth_df = spop.generate(len(df))
        synth_df2 = synth_df.iloc[:10]
        synth_df.to_csv("C:/finalproject/myproject/pybo/synth_dir/" + obj + ".csv", index= False)
        return render_template('synth_generate.html', df=synth_df2, dtypes=synth_df.dtypes)
        # return render_template('synth_generate.html', df=df, dtypes=dtypes)


# json 생성
@bp.route('/distribution', methods=['GET', 'POST'])
def distribution():
    obj = g.user.username
    original_data = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
    synth_data = pd.read_csv("C:/finalproject/myproject/pybo/synth_dir/" + obj + ".csv")
    cate_col = []
    for i in range(0, len(original_data.columns)):
        if original_data.dtypes[i] != 'object':
            cate_col.append(original_data.columns[i])
    fig = plt.figure()
    rows = 1
    cols = len(cate_col)
    for i in range(0, cols):
        plt.subplot(1, cols, i + 1)
        plt.hist(synth_data[cate_col[i]], alpha=0.3)
        plt.hist(original_data[cate_col[i]], alpha=0.3)
        plt.title(cate_col[i])
        plt.legend(['synth', 'origin'])
        plt.savefig('C:/finalproject/myproject/pybo/static/img_dir/' + obj + 'dis.png')

    return render_template('distribution.html', image_file='/img_dir/'+ obj + 'dis.png')





