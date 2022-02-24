import time
from flask import Blueprint, render_template, url_for, request, session, jsonify, g
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
# warning
import warnings

warnings.filterwarnings('ignore')
from flask import send_file
from flask import send_from_directory
from pybo.views.auth_views import login_required
from operator import is_not
from functools import partial
from sklearn.linear_model import LinearRegression
import PIL
from PIL import Image, ImageDraw, ImageFont
import os
import pymysql
import datetime
import boto3
import io

bp = Blueprint('main', __name__, url_prefix='/')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.debug = True
ALLOWED_EXTENSIONS = set(['txt', 'csv'])
app.config.update(
    PERMANENT_SESSION_LIFETIME=600
)

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
    obj = g.user.username
    file = "/home/ubuntu/projects/FlaskProject//home/ubuntu/projects/FlaskProject/uploads/" + obj + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass

    file2 = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".json"
    try:
        os.remove(file2)
    except OSError:
        pass

    list2 = ['origincorr', 'originreg', 'synthcorr', 'synthreg', 'dis']
    for i in range(0, len(list2)):
        file3 = "/home/ubuntu/projects/FlaskProject/static/img_dir/" + obj + list2[i] + ".png"
        try:
            os.remove(file3)
        except OSError:
            pass
    return render_template('upload.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 파일 업로드 처리
@bp.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
        obj = g.user.username

            #  ff = pd.DataFrame(data = f)
        fff = pd.read_csv("/home/ubuntu/projects/FlaskProject/uploads/" + obj + '.csv')
        cate_col = []
        for i in range(0, len(fff.columns)):
            if fff.dtypes[i] != 'object':
                cate_col.append(fff.columns[i])

        std = fff[cate_col].std()
        mean = fff[cate_col].mean()
        a = []

        for i in range(len(mean)):
            line = []
            for j in range(1):
                line.append(mean[i] - (3 * std[i]))
                line.append(mean[i] + (3 * std[i]))
            a.append(line)

        k = 0
        delete = []

        for i in fff[cate_col]:
            for j in range(len(fff[cate_col])):
                if fff[cate_col][i][j] < a[k][0] or fff[cate_col][i][j] > a[k][1]:
                    delete.append(j)
            k += 1

        my_set = set(delete)  # 집합set으로 변환
        delete = list(my_set)  # list로 변환

        for i in delete:
            fff.drop(i, axis=0, inplace=True)

        df_info = fff

        df_info3 = df_info.iloc[0:10]

        df_col = []

        for i in range(0, len(fff.columns)):
            df_col.append(fff.columns[i])

        count = {}
        for i in range(0, len(df_col)):
            count[i] = df_col[i]

        s3 = boto3.resource('s3')
        bucket_name = 'origindir'
        bucket = s3.Bucket(name=bucket_name)

        local_file = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + '.csv'
        obj_file = str(obj) + '.csv'
        bucket.upload_file(local_file, obj_file)

        file = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".csv"
        try:
            os.remove(file)
        except OSError:
            pass

        return render_template('upload2.html', tables=[df_info3.to_html()], titles=[''], count=count)


# json 생성 및 재현데이터 생성
@bp.route('/partsynth_generate', methods=['GET', 'POST'])
def partsynth_generate():
    obj = g.user.username

    s3 = boto3.resource('s3')
    bucket_name = 'origindir'
    bucket = s3.Bucket(name=bucket_name)
    aa = obj + '.csv'
    fff = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))

    df_col2 = []

    for i in range(0, len(fff.columns)):
        df_col2.append(fff.columns[i])
    df_type2 = []
    for i in range(0, len(fff.columns)):
        if (fff.dtypes[i] == 'int64'):
            df_type2.append('int')
        elif (fff.dtypes[i] == 'float64'):
            df_type2.append('float')
        else:
            df_type2.append('category')

    to_json = {df_col2[0]: df_type2[0]}
    # 이렇게 dict로 주지 않으면 list형식으로 들어감 ;
    for i in range(1, len(fff.columns)):
        to_json[df_col2[i]] = df_type2[i]
    with open("/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".json", 'w') as f:
        json.dump(to_json, f)

    # df = pd.read_csv("/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".csv")
    with open("/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".json", 'r') as f:
        dtypes = json.load(f)
    columns = list(dtypes.keys())

    fff.to_csv("/home/ubuntu/projects/FlaskProject/uploads/" + obj + '.csv')

    rrf = pd.read_csv("/home/ubuntu/projects/FlaskProject/uploads/" + obj + '.csv', header=None, skiprows=1,
                      names=columns).astype(dtypes)

    file = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass

    # rrf = rrf.iloc[:,1:]
    # rrf.apply(pd.to_numeric, errors='coerce')

    spop = Synthpop()
    spop.fit(rrf, dtypes)

    synth_df = spop.generate(len(rrf))

    df_col = []
    list1 = []
    for i in range(0, len(rrf.columns)):
        df_col.append(rrf.columns[i])
    if request.method == 'POST':
        for i in range(0, len(df_col)):
            list1.append(request.form.get(df_col[i]))
    list2 = list(filter(None.__ne__, list1))

    df2 = rrf.drop(list2, axis=1)

    add_divide()

    synth_df2 = synth_df[list2]
    result = pd.concat([synth_df2, df2], axis=1)

    df = fff.iloc[:10]
    result2 = result.iloc[:10]
    result.to_csv("/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv", index=False)

    return render_template('partsynth_generate.html', tables=[df.to_html()], titles=[''], tables2=[result2.to_html()],
                           titles2=[''])


@bp.route('/syn_store2', methods=['GET', 'POST'])
@login_required
def syn_store2():
    obj = g.user.username
    db = pymysql.connect(host="master.ckekx9n1eyul.ap-northeast-2.rds.amazonaws.com", user="admin",
                         passwd="!dldirl7310", db="test", charset="utf8")
    cur = db.cursor()

    sql = "SELECT * from testtable where id = (%s)"
    cur.execute(sql, (obj))
    data_list = cur.fetchall()

    db.commit()
    db.close()
    count = {}
    for i in range(0, len(data_list)):
        count[i] = data_list[i][0]
    # list2 = list(filter(None.__ne__, list1))
    # df_info = df_info.drop(list2, axis=1)

    return render_template('syn_store2.html', count=count)


@bp.route('/hello11', methods=['GET', 'POST'])
def hello11():
    obj = g.user.username
    db = pymysql.connect(host="master.ckekx9n1eyul.ap-northeast-2.rds.amazonaws.com", user="admin",
                         passwd="!dldirl7310", db="test", charset="utf8")
    cur = db.cursor()

    sql = "SELECT * from testtable where id = (%s)"
    cur.execute(sql, (obj))
    data_list = cur.fetchall()

    db.commit()
    db.close()

    count = []
    for i in range(0, len(data_list)):
        count.append(data_list[i][0])

    if request.method == 'POST':
        val = request.form

    list1 = list(val.keys())

    list_a = list(map(int, list1))
    count2 = []
    for i in list_a:
        count2.append(count[i])
    s3 = boto3.resource('s3')
    bucket_name = 'synthdir'
    bucket = s3.Bucket(bucket_name)
    for j in range(0, len(count2)):
        obj_file = count2[j] + '.csv'
        save_file = '/home/ubuntu/projects/FlaskProject/synth_dir/' + count2[j] + '.csv'
        bucket.download_file(obj_file, save_file)

    for k in range(0, len(count2)):
        syn = "/home/ubuntu/projects/FlaskProject/synth_dir/" + str(count2[0]) + ".csv"
        result = send_file(syn, as_attachment=True)
        return result

    for l in range(0, len(count2)):
        file = "/home/ubuntu/projects/FlaskProject/uploads/" + count2[l] + ".csv"
        try:
            os.remove(file)
        except OSError:
            pass

    return render_template('test.html')

    # return str(list1)


index_add_counter = []


def syn_down(filename):
    syn = "/home/ubuntu/projects/FlaskProject/synth_dir/" + filename + ".csv"
    return send_file(syn, as_attachment=True)


def add_divide():
    global index_add_counter
    index_add_counter.clear()

    today = datetime.datetime.now()

    index_add_counter.append(today.year)
    index_add_counter.append(today.month)
    index_add_counter.append(today.day)
    index_add_counter.append(today.hour)
    index_add_counter.append(today.minute)
    index_add_counter.append(today.microsecond)


# 유사도 측정
@bp.route('/distribution', methods=['GET', 'POST'])
def distribution():
    obj = g.user.username
    s3 = boto3.resource('s3')
    bucket_name = 'origindir'
    bucket = s3.Bucket(name=bucket_name)
    aa = obj + '.csv'
    original_data = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))

    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv")
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
        plt.savefig('/home/ubuntu/projects/FlaskProject/static/img_dir/' + obj + 'dis.png')

    plt.close()
    return render_template('distribution.html', image_file='/img_dir/' + obj + 'dis.png')


# 회귀분석 -
@bp.route('/regression', methods=['GET', 'POST'])
def regression():
    obj = g.user.username

    s3 = boto3.resource('s3')
    bucket_name = 'origindir'
    bucket = s3.Bucket(name=bucket_name)
    aa = obj + '.csv'
    original_data = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))

    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv")
    cate_col = []
    for i in range(0, len(original_data.columns)):
        if original_data.dtypes[i] != 'object':
            cate_col.append(original_data.columns[i])
    original_data = original_data[cate_col]
    # original_data = original_data.iloc[:, 1:]
    synth_data = synth_data[cate_col]
    lr = LinearRegression()
    etc = []
    for i in range(0, len(cate_col)):
        if (i == 0):
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

    target_image = Image.open('/home/ubuntu/projects/FlaskProject/static/img_dir/baseimg.png')
    draw = ImageDraw.Draw(target_image)
    font = ImageFont.load_default()
    draw.text((10, 10), list, fill="black", font=font)
    target_image.save('/home/ubuntu/projects/FlaskProject/static/img_dir/' + obj + 'originreg.png')
    target_image.close()

    target_image2 = Image.open('/home/ubuntu/projects/FlaskProject/static/img_dir/baseimg2.png')
    font2 = ImageFont.load_default()
    draw2 = ImageDraw.Draw(target_image2)
    draw2.text((10, 10), list2, fill="black", font=font2)
    target_image2.save('/home/ubuntu/projects/FlaskProject/static/img_dir/' + obj + 'synthreg.png')
    target_image2.close()

    return render_template('regression.html', origin_file='/img_dir/' + obj + 'originreg.png',
                           synth_file='/img_dir/' + obj + 'synthreg.png')


# 상관관계 분석 - 수치형만
@bp.route('/correlation', methods=['GET', 'POST'])
def correlation():
    obj = g.user.username
    plt.close()
    # original_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".csv")
    s3 = boto3.resource('s3')
    bucket_name = 'origindir'
    bucket = s3.Bucket(name=bucket_name)
    aa = obj + '.csv'
    original_data = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))

    # original_data = original_data.iloc[:, 1:]
    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv")
    corr_df = original_data.corr()
    corr_df = corr_df.apply(lambda x: round(x, 2))

    ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('/home/ubuntu/projects/FlaskProject/static/img_dir/' + obj + 'origincorr.png')
    plt.close()
    corr_df2 = synth_data.corr()
    corr_df2 = corr_df2.apply(lambda x: round(x, 2))

    ax2 = sns.heatmap(corr_df2, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('/home/ubuntu/projects/FlaskProject/static/img_dir/' + obj + 'synthcorr.png')
    plt.close()

    return render_template('correlation.html', synth_file='/img_dir/' + obj + 'synthcorr.png',
                           origin_file='/img_dir/' + obj + 'origincorr.png')


@bp.route('/syn_store', methods=['GET', 'POST'])
def syn_store():
    obj = g.user.username
    db = pymysql.connect(host="master.ckekx9n1eyul.ap-northeast-2.rds.amazonaws.com", user="admin",
                         passwd="!dldirl7310", db="test", charset="utf8")
    cur = db.cursor()

    sql = 'insert into testtable (csvname, id, gen_time) values(%s,%s,%s)'
    cur.execute(sql, (obj + str(index_add_counter), obj, datetime.datetime.now()))

    sql = "SELECT * from testtable where id = (%s)"
    cur.execute(sql, (obj))
    data_list = cur.fetchall()

    db.commit()
    db.close()
    s3 = boto3.resource('s3')

    bucket_name = 'synthdir'
    bucket = s3.Bucket(bucket_name)

    local_file = "/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv"
    obj_file = obj + str(index_add_counter) + '.csv'
    bucket.upload_file(local_file, obj_file)

    file = "/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass
    stst = obj + str(index_add_counter)
    return render_template('syn_store.html', stst = stst)


@bp.route('/hello3')
def hello_pybo3():
    obj = g.user.username

    file = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass

    syn = "/home/ubuntu/projects/FlaskProject/synth_dir/" + obj + str(index_add_counter) + ".csv"

    file2 = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + ".json"
    try:
        os.remove(file2)
    except OSError:
        pass

    file3 = "/home/ubuntu/projects/FlaskProject/uploads/" + obj + "%.png"
    try:
        os.remove(file3)
    except OSError:
        pass

    return send_file(syn, as_attachment=True)


import urllib.parse


def convert(input):
    # Converts unicode to string
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, str):
        return input.encode('utf-8')
    else:
        return input

@bp.route("/target_endpoint", methods = ['GET', 'POST'])
def target():
    if request.method == 'POST':
        # You could do any information passing here if you want (i.e Post or Get request)
        some_data = "Here's some example data"
        some_data = urllib.parse.quote(
            convert(some_data))  # urllib2 is used if you have fancy characters in your data like "+"," ", or "="
        # This is where the loading screen will be.
        # ( You don't have to pass data if you want, but if you do, make sure you have a matching variable in the html i.e {{my_data}} )
        s3 = boto3.resource('s3')

        bucket_name = 'origindir'
        bucket = s3.Bucket(bucket_name)
        obj = g.user.username
        f = request.files['file']
        if f and allowed_file(f.filename):
            f.save("/home/ubuntu/projects/FlaskProject/uploads/" + obj + '.csv')
        else:
            return render_template('extension_error.html')


        return render_template('loading.html', my_data = some_data)