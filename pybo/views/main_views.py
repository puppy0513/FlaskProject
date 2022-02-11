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
import seaborn as sns
import numpy as np
#warning
import warnings
warnings.filterwarnings('ignore')
from flask import send_file
from flask import send_from_directory
from pybo.views.auth_views import login_required
from operator import is_not
from functools import partial
from sklearn.linear_model import  LinearRegression
import PIL
from PIL import Image, ImageDraw, ImageFont
import os
import pymysql



bp = Blueprint('main', __name__, url_prefix='/')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.debug = True

from pybo.models import Question
# from datasets.adult import df, dtypes
# from pybo.synthpop.datasets.adult import df, dtypes

global obj

@bp.route('/manual')
def manual():
    return render_template('manual.html')


@bp.route('/index')
def index():
    return redirect(url_for('question._list'))


@bp.route('/')
def main():
    return render_template('main.html')

# 업로드 HTML 렌더링
@bp.route('/upload')
@login_required
def render_file():

    return render_template('upload.html')




# /home/ubuntu/projects/myproject/pybo/uploads
# 파일 업로드 처리
@bp.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # f = pd.DataFrame(data = request.files['file'])
        # 저장할 경로 + 파일명

        obj = g.user.username
        #  ff = pd.DataFrame(data = f)
        f.save("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + '.csv')
        # fff = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + '.csv', encoding='CP949')
        fff = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + '.csv')
        # return f.to_html()
        df_info = fff
        df_col = []
        for i in range(0, len(df_info.columns)):
            df_col.append(df_info.columns[i])


        df2 = df_info.iloc[:10]

        return render_template('upload2.html', tables=[df2.to_html()], titles=[''])


# json 생성
@bp.route('/to_json', methods=['GET', 'POST'])
def to_json():
        obj = g.user.username
        df_info = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv")
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
        with open("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".json", 'w') as f:
            json.dump(to_json, f)
        
        return render_template('to_json.html', df=df_info, df_col=df_col)


# json 생성
@bp.route('/to_json_part', methods=['GET', 'POST'])
def to_json_part():
    obj = g.user.username
    df_info = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv")
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
    with open("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".json", 'w') as f:
        json.dump(to_json, f)

    count = {}
    for i in range(0, len(df_col)):
        count[i] = df_col[i]

    return render_template('to_json_part.html', df_col=df_col, count = count)

# json 생성2
@bp.route('/to_json_part2', methods=['GET', 'POST'])
def to_json_part2():
    obj = g.user.username
    df_info = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv")
    df_col = []
    list1 = []
    for i in range(0, len(df_info.columns)):
        df_col.append(df_info.columns[i])
    if request.method == 'POST':
        for i in range(0, len(df_col)):
            list1.append(request.form.get(df_col[i]))
    list2 = list(filter(None.__ne__, list1))
    df_info = df_info.drop(list2, axis=1)
    df_info.to_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + '.csv')
    df_info2 = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + '.csv')
    df_info2 = df_info2.iloc[:,1:]
    df_info2.to_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + '.csv')
    df_col2 = []

    for i in range(0, len(df_info2.columns)):
        df_col2.append(df_info2.columns[i])
    df_type2 = []
    for i in range(0, len(df_info2.columns)):
        if (df_info2.dtypes[i] == 'int64'):
            df_type2.append('int')
        elif (df_info2.dtypes[i] == 'float64'):
            df_type2.append('float')
        else:
            df_type2.append('category')

    to_json = {df_col2[0]: df_type2[0]}
    # 이렇게 dict로 주지 않으면 list형식으로 들어감 ;
    for i in range(1, len(df_info2.columns)):
        to_json[df_col2[i]] = df_type2[i]
    with open("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".json", 'w') as f:
        json.dump(to_json, f)
    count = {}
    for i in range(0, len(df_col2)):
        count[i] = df_col2[i]

    return render_template('to_json_part2.html', df_col=df_col2, count = count)


# json 생성 및 재현데이터 생성
@bp.route('/synth_generate', methods=['GET', 'POST'])
def synth_generate():
        obj = g.user.username

        # df = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
        with open("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".json", 'r') as f:
            dtypes = json.load(f)
        columns = list(dtypes.keys())

        df = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv", header=None, skiprows = 1, names=columns).astype(dtypes)
        # 헤더가 있는 경우 -> skip

        df.apply(pd.to_numeric, errors='coerce')

        spop = Synthpop()
        spop.fit(df, dtypes)

        synth_df = spop.generate(len(df))
        synth_df2 = synth_df.iloc[:10]
        df2 = df.iloc[:10]
        synth_df.to_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/synth_dir/" + obj + ".csv", index= False)
        return render_template('synth_generate.html', tables=[df2.to_html()], titles=[''], tables2=[synth_df2.to_html()], titles2=[''])
        # return render_template('synth_generate.html', df=df, dtypes=dtypes)

# json 생성 및 재현데이터 생성
@bp.route('/partsynth_generate', methods=['GET', 'POST'])
def partsynth_generate():
        obj = g.user.username

        # df = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
        with open("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".json", 'r') as f:
            dtypes = json.load(f)
        columns = list(dtypes.keys())

        df = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv", header=None, skiprows = 1, names=columns).astype(dtypes)
        # 헤더가 있는 경우 -> skip
        df.apply(pd.to_numeric, errors='coerce')

        spop = Synthpop()
        spop.fit(df, dtypes)

        synth_df = spop.generate(len(df))
        df_col = []
        for i in range(0, len(df.columns)):
            df_col.append(df.columns[i])
        list1 = []
        if request.method == 'POST':
            for i in range(0, len(df_col)):
                list1.append(request.form.get(df_col[i]))
        list2 = list(filter(None.__ne__, list1))
        df2 = df.drop(list2, axis=1)


        synth_df2 = synth_df[list2]
        result = pd.concat([df2, synth_df2], axis=1)
        df = df.iloc[:10]
        result2 = result.iloc[:10]
        result.to_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/synth_dir/" + obj + ".csv", index= False)
        return render_template('partsynth_generate.html', tables=[df.to_html()], titles=[''], tables2=[result2.to_html()], titles2=[''])



# 유사도 측정
@bp.route('/distribution', methods=['GET', 'POST'])
def distribution():
    obj = g.user.username
    original_data = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv")
    original_data = original_data.iloc[:,1:]
    synth_data = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/synth_dir/" + obj + ".csv")
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
        plt.savefig('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/' + obj + 'dis.png')

    plt.close()
    return render_template('distribution.html', image_file='/img_dir/'+ obj + 'dis.png')


# 회귀분석 -
@bp.route('/regression', methods=['GET', 'POST'])
def regression():
    obj = g.user.username
    original_data = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv")
    original_data = original_data.iloc[:, 1:]
    synth_data = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/synth_dir/" + obj + ".csv")
    cate_col = []
    for i in range(0, len(original_data.columns)):
        if original_data.dtypes[i] != 'object':
            cate_col.append(original_data.columns[i])
    original_data = original_data[cate_col]
    # original_data = original_data.iloc[:, 1:]
    synth_data = synth_data[cate_col]
    lr = LinearRegression()
    etc = []
    for i in range(0,  len(cate_col)):
        if(i ==0 ):
            standard = cate_col[i]
        else:
            etc.append(cate_col[i])

    Y = original_data[standard].values
    X = original_data.loc[:, etc].values
    import statsmodels.api as sm
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    # results.summary()

    etc2 = []
    for i in range(0, len(cate_col)):
        if (i == 0):
            standard2 = cate_col[i]
        else:
            etc2.append(cate_col[i])

    Y2 = synth_data[standard2].values
    X2 = synth_data.loc[:, etc2].values
    results2 = sm.OLS(Y2, sm.add_constant(X2)).fit()
    list = str(results.summary())
    list2 = str(results2.summary())


    target_image = Image.open('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/baseimg.png')
    draw =ImageDraw.Draw(target_image)
    font = ImageFont.truetype("arial.ttf", 15)
    draw.text((10,10),list,fill="black",font=font)
    target_image.save('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/' + obj + 'originreg.png')
    target_image.close()


    target_image2 = Image.open('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/baseimg2.png')
    font2 = ImageFont.truetype("arial.ttf", 15)
    draw2 = ImageDraw.Draw(target_image2)
    draw2.text((10, 10), list2, fill="black", font=font2)
    target_image2.save('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/' + obj + 'synthreg.png')
    target_image2.close()


    return render_template('regression.html', origin_file='/img_dir/'+ obj + 'originreg.png', synth_file='/img_dir/'+ obj + 'synthreg.png')


# 상관관계 분석 - 수치형만
@bp.route('/correlation', methods=['GET', 'POST'])
def correlation():
    obj = g.user.username
    plt.close()
    original_data = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv")
    original_data = original_data.iloc[:, 1:]
    synth_data = pd.read_csv("C:/Work/최종프로젝트/flask/FlaskProject/pybo/synth_dir/" + obj + ".csv")
    corr_df = original_data.corr()
    corr_df = corr_df.apply(lambda x: round(x, 2))

    ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/' + obj + 'origincorr.png')
    plt.close()
    corr_df2 = synth_data.corr()
    corr_df2 = corr_df2.apply(lambda x: round(x, 2))

    ax2 = sns.heatmap(corr_df2, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('C:/Work/최종프로젝트/flask/FlaskProject/pybo/static/img_dir/' + obj + 'synthcorr.png')
    plt.close()

    return render_template('correlation.html', synth_file='/img_dir/'+ obj + 'synthcorr.png', origin_file='/img_dir/'+ obj + 'origincorr.png')



@bp.route('/hello3')
def hello_pybo3():
    obj = g.user.username

    file = "C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".csv"
    if os.path.isfile(file):
        os.remove(file)
    path = "C:/Work/최종프로젝트/flask/FlaskProject/pybo/synth_dir/" + obj + ".csv"
    file2 = "C:/Work/최종프로젝트/flask/FlaskProject/pybo/uploads/" + obj + ".json"
    if os.path.isfile(file2):
        os.remove(file2)





    return send_file(path, as_attachment=True)






# -------------------------------------- 여기부터는 연습용------------------------------------
@bp.route('/hello',  methods=['GET','POST'])
def hello_pybo():
    obj = g.user.username
    return obj

'''
# POST 형식으로 HTML 데이터 가져오기 -> 정제
@bp.route('/hello2',  methods=['GET','POST'])
def hello_pybo2():
    obj = g.user.username
    df_info = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
    df_col = []
    list1 = []
    for i in range(0, len(df_info.columns)):
        df_col.append(df_info.columns[i])
    if request.method == 'POST':
        for i in range(0, len(df_col)):
            list1.append(request.form.get(df_col[i]))
    list2 = list(filter(None.__ne__, list1))
    return render_template('main.html', list = list2)
'''
@bp.route('/hello2',  methods=['GET','POST'])
def hello_pybo2():
    obj = g.user.username
    original_data = pd.read_csv("C:/finalproject/myproject/pybo/uploads/" + obj + ".csv")
    original_data = original_data.iloc[:, 1:]
    synth_data = pd.read_csv("C:/finalproject/myproject/pybo/synth_dir/" + obj + ".csv")
    cate_col = []
    for i in range(0, len(original_data.columns)):
        if original_data.dtypes[i] != 'object':
            cate_col.append(original_data.columns[i])
    original_data = original_data[cate_col]
    return render_template('regression.html', list=original_data)




