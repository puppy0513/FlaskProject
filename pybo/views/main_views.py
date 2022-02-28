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
import pyemd
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
import matplotlib.patches as patches

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
    file = "/home/ubuntu/projects/FlaskProject/pybo//home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass

    file2 = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".json"
    try:
        os.remove(file2)
    except OSError:
        pass

    list2 = ['origincorr', 'originreg', 'synthcorr', 'synthreg', 'dis']
    for i in range(0, len(list2)):
        file3 = "/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/" + obj + list2[i] + ".png"
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
        fff = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + '.csv')
        cate_col = []
        for i in range(0, len(fff.columns)):
            if fff.dtypes[i] != 'object':
                cate_col.append(fff.columns[i])

        sensitive_col = []
        for i in range(0, len(fff.columns)):
            if fff.dtypes[i] == 'object':
                sensitive_col.append(fff.columns[i])


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

        sensitive_count = {}
        for i in range(0, len(sensitive_col)):
            sensitive_count[i+100] = sensitive_col[i]

        identifier_count = {}
        for i in range(0, len(cate_col)):
            identifier_count[i+200] = cate_col[i]


        s3 = boto3.resource('s3')
        bucket_name = 'origindir'
        bucket = s3.Bucket(name=bucket_name)

        local_file = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + '.csv'
        obj_file = str(obj) + '.csv'
        bucket.upload_file(local_file, obj_file)

        file = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".csv"
        try:
            os.remove(file)
        except OSError:
            pass

        return render_template('upload2.html', tables=[df_info3.to_html()], titles=[''], count=count,
                               sensitive_count = sensitive_count, identifier_count = identifier_count
                               )


@bp.route("/target_endpoint2", methods=['GET', 'POST'])
def target2():
    obj = g.user.username
    s3 = boto3.resource('s3')
    bucket_name = 'origindir'
    bucket = s3.Bucket(name=bucket_name)
    aa = obj + '.csv'
    fff = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))
    first.clear()
    risk_category.clear()
    risk_num.clear()
    df_col = []

    for i in range(0, len(fff.columns)):
        df_col.append(fff.columns[i])
    list1 = []
    list_sen = []
    list_iden = []

    if request.method == 'POST':
        for i in range(0, len(df_col)):
            list1.append(request.form.get(df_col[i]))
            list_sen.append(request.form.get(df_col[i] + 'a'))
            list_iden.append(request.form.get(df_col[i] + 'b'))

    list2 = list(filter(None.__ne__, list1))
    list_sen_1 = list(filter(None.__ne__, list_sen))
    list_iden_1 = list(filter(None.__ne__, list_iden))

    first.append(list2)
    risk_category.append(list_sen_1)
    risk_num.append(list_iden_1)

    # You could do any information passing here if you want (i.e Post or Get request)
    some_data = "Here's some example data"
    some_data = urllib.parse.quote(
        convert(some_data))  # urllib2 is used if you have fancy characters in your data like "+"," ", or "="
    # This is where the loading screen will be.
    # ( You don't have to pass data if you want, but if you do, make sure you have a matching variable in the html i.e {{my_data}} )

    return render_template('loading2.html', my_data=some_data)



first = []

''''
@bp.route("/oh", methods=['GET', 'POST'])
def oh():
    if request.method == 'POST':
        df_col = []
        obj = g.user.username

        s3 = boto3.resource('s3')
        bucket_name = 'origindir'
        bucket = s3.Bucket(name=bucket_name)
        aa = obj + '.csv'
        fff = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))
        for i in range(0, len(fff.columns)):
            df_col.append(fff.columns[i])

        list1 = []
        list_sen = []
        list_iden = []
        for i in range(0, len(df_col)):
            list1.append(request.form.get(df_col[i]))
            list_sen.append(request.form.get(df_col[i]+'a'))
            list_iden.append(request.form.get(df_col[i]+'b'))


        list2 = list(filter(None.__ne__, list1))
        list_sen_1 = list(filter(None.__ne__, list_sen))
        list_iden_1 = list(filter(None.__ne__, list_iden))
        first.append(list2)
        risk_category.append(list_sen_1)
        risk_num.append(list_iden_1)
        return render_template('oh.html', list2=list2, list_sen = list_sen_1, list_iden = list_iden_1)
'''


# json 생성 및 재현데이터 생성
@bp.route('/partsynth_generate', methods=['GET', 'POST'])
def partsynth_generate():
    # -------------------------------------- 여기부터는 T근접성 함수------------------------------------

    '''
    데이터 프레임의 파티션에 대해 모든 열의 스팬
    (숫자 열의 경우 max-min, 범주형 열의 경우 서로 다른 값의 수)을
    반환하는 함수를 구현합니다.
    '''

    def get_spans(df, partition, scale=None):
        """
        :param        df: the dataframe for which to calculate the spans
        :param partition: the partition for which to calculate the spans
        :param     scale: if given, the spans of each column will be divided
                          by the value in `scale` for that column
        :        returns: The spans of all columns in the partition
        """
        spans = {}
        for column in df.columns:
            if column in cate_col:
                span = len(df[column][partition].unique())
            else:
                span = df[column][partition].max() - df[column][partition].min()
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    # print('모든 열에 대한 빈도 수 구하기')
    # print(full_spans)

    '''
    데이터 프레임, 파티션 및 열을 사용하고 지정된 파티션 두 개를 반환하는 
    분할 함수를 구현하여 열 값이 중위수보다 크거나 같은 모든 행이 
    다른 파티션에 있도록 합니다.
    '''

    def split(df, partition, column):
        """
        :param        df: The dataframe to split
        :param partition: The partition to split
        :param    column: The column along which to split
        :        returns: A tuple containing a split of the original partition
        """
        dfp = df[column][partition]
        if column in cate_col:
            values = dfp.unique()
            lv = set(values[:len(values) // 2])
            rv = set(values[len(values) // 2:])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)

    '''
    이제 모든 도우미 함수가 적용되었으므로 위에서 설명한 파티션 알고리즘을 구현할 수 있습니다.
    위에서 설명한 파티션 알고리즘을 생성하는 파티션에 대해 k-익명 기준을 사용하여 구현합니다.
    '''

    def is_k_anonymous(df, partition, sensitive_column, k=3):
        """
        :param               df: The dataframe on which to check the partition.
        :param        partition: The partition of the dataframe to check.
        :param sensitive_column: The name of the sensitive column
        :param                k: The desired k
        :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
        """
        if len(partition) < k:
            return False
        return True

    def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
        """
        :param               df: The dataframe to be partitioned.
        :param  feature_columns: A list of column names along which to partition the dataset.
        :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
        :param            scale: The column spans as generated before.
        :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
        :returns               : A list of valid partitions that cover the entire dataframe.
        """
        finished_partitions = []
        partitions = [df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = get_spans(df[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = split(df, partition, column)
                if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    '''
    단순하게 유지하기 위해 먼저 분할을 적용할 데이터 집합에서 열을 두 개만 선택합니다. 
    따라서 결과를 쉽게 확인/시각화하고 실행 속도를 높일 수 있습니다
    (전체 데이터 세트에서 실행할 경우 간단한 알고리즘이 몇 분 정도 걸릴 수 있음).
    '''

    # 우리는 "gender"을 민감한(sensitive) 속성으로 사용하여 데이터셋의 두 열에
    # 분할 방법을 적용한다.

    # 생성된 파티션 수를 가져옵니다.

    # print('생성된 파티션 수를 가져옵니다.')
    # print(len(finished_partitions))

    '''
    생성된 파티션을 시각화해 보겠습니다! 
    이를 위해 두 열을 따라 파티션의 직사각형 경계를 구하는 함수를 작성할 것입니다. 
    그런 다음 이러한 직장을 그래프로 표시하여 분할 함수가 데이터 집합을 어떻게 분할하는지 확인할 수 있습니다. 
    플롯팅하기 위해 선택한 두 개의 열을 따라서만 파티션을 수행하면 
    결과 직장이 겹치지 않고 전체 데이터 세트를 덮을 수 있습니다.
    '''

    def build_indexes(df):
        indexes = {}
        for column in cate_col:
            values = sorted(df[column].unique())
            indexes[column] = {x: y for x, y in zip(values, range(len(values)))}
        return indexes

    def get_coords(df, column, partition, indexes, offset=0.1):
        if column in cate_col:
            sv = df[column][partition].sort_values()
            l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]] + 1.0
        else:
            sv = df[column][partition].sort_values()
            next_value = sv[sv.index[-1]]
            larger_values = df[df[column] > next_value][column]
            if len(larger_values) > 0:
                next_value = larger_values.min()
            l = sv[sv.index[0]]
            r = next_value
        # we add some offset to make the partitions more easily visible
        l -= offset
        r += offset
        return l, r

    def get_partition_rects(df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
        rects = []
        for partition in partitions:
            xl, xr = get_coords(df, column_x, partition, indexes, offset=offsets[0])
            yl, yr = get_coords(df, column_y, partition, indexes, offset=offsets[1])
            rects.append(((xl, yl), (xr, yr)))
        return rects

    def get_bounds(df, column, indexes, offset=1.0):
        if column in cate_col:
            return 0 - offset, len(indexes[column]) + offset
        return df[column].min() - offset, df[column].max() + offset

    # 생성한 모든 파티션의 경계 직장을 계산합니다.

    # print(rects[:10])

    def plot_rects(df, ax, rects, column_x, column_y, edgecolor='black', facecolor='none'):
        for (xl, yl), (xr, yr) in rects:
            ax.add_patch(
                patches.Rectangle((xl, yl), xr - xl, yr - yl, linewidth=1, edgecolor=edgecolor, facecolor=facecolor,
                                  alpha=0.5))
        ax.set_xlim(*get_bounds(df, column_x, indexes))
        ax.set_ylim(*get_bounds(df, column_y, indexes))
        ax.set_xlabel(column_x)
        ax.set_ylabel(column_y)

    def agg_categorical_column(series):
        # workearound here
        series.astype('category')
        return [','.join(set(series))]

    def agg_numerical_column(series):
        return [series.mean()]

    def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
        aggregations = {}
        for column in feature_columns:
            if column in cate_col:
                aggregations[column] = agg_categorical_column
            else:
                aggregations[column] = agg_numerical_column
        rows = []
        for i, partition in enumerate(partitions):
            if i % 100 == 1:
                print("Finished {} partitions...".format(i))
            if max_partitions is not None and i > max_partitions:
                break
            grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
            # print(grouped_columns.iloc[0])
            sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column: 'count'})
            values = {}
            for name, val in grouped_columns.items():
                values[name] = val[0]
            for sensitive_value, count in sensitive_counts[sensitive_column].items():
                if count == 0:
                    continue
                values.update({
                    sensitive_column: sensitive_value,
                    'count': count,

                })
                rows.append(values.copy())
        return pd.DataFrame(rows)

    '''

    print('범주형 열 및 민감 속성을 사용하여 결과 데이터 프레임을 정렬합니다.')
    print(dfn)
    dfn.to_csv('syn_dft.scv', index = False)




    l-다양성 구현
    l-다양성을 구현하기 위해 다음과 같은 작업을 수행할 수 있습니다.

    is_valid 함수를 수정하여 주어진 파티션의 크기를 확인할 뿐만 아니라 
    파티션의 중요한 특성 값이 충분히 다양한지 확인하십시오.
    분할 함수를 수정하여 다양한 분할을 생성합니다(가능한 경우).
    지정된 파티션에 중요한 특성의 다른 값이 l개 이상 포함되어 있으면 
    True를 반환하고 그렇지 않으면 False를 반환하는 검증기 함수를 구현합니다.
    -> 여기서는 열 2개이므로 l=2
    '''

    def diversity(df, partition, column):
        return len(df[column][partition].unique())

    def is_l_diverse(df, partition, sensitive_column, l=2):
        """
        :param               df: The dataframe for which to check l-diversity
        :param        partition: The partition of the dataframe on which to check l-diversity
        :param sensitive_column: The name of the sensitive column
        :param                l: The minimum required diversity of sensitive attribute values in the partition
        """
        return diversity(df, partition, sensitive_column) >= l

    # 이 방법을 데이터에 적용하고 결과가 어떻게 변하는지 봅시다.

    '''
    t-closiness 구현
    보시다시피, 값 다양성이 낮은 영역의 경우, l-diverse 방법은 
    중요한 속성의 한 값에 대한 매우 많은 항목과 다른 값에 대한 하나의 항목만 
    포함하는 파티션을 생성합니다. 이는 데이터셋에 있는 사람에 대한 
    "타당한 거부 가능성"이 있는 반면(모든 사람이 "아웃라이어"가 될 수 있기 때문에)
     적수는 여전히 그 사람의 속성 가치에 대해 매우 확실할 수 있다.

    t-유효성은 지정된 파티션에서 중요한 속성 값의 분포가 전체 데이터 세트의 값 분포와
     유사하도록 함으로써 이 문제를 해결합니다.

    파티션이 충분히 다양하면 True를 반환하고 그렇지 않으면 False를 반환하는 
    is_valid 함수의 버전을 구현합니다. 
    다양성을 측정하기 위해 전체 데이터 세트에 대한 민감한 속성의 경험적 확률 분포와 
    파티션에 대한 분포 사이의 콜모고로프-스미르노프 거리를 계산한다. 
    힌트: 콜모고로프-스미르노프 거리는 두 분포 사이의 최대 거리입니다. 
    중요한 특성은 범주형 값이라고 가정할 수 있습니다.
    '''

    # generate the global frequencies for the sensitive column

    def t_closeness(df, partition, column, global_freqs):
        total_count = float(len(partition))
        d_max = None
        group_counts = df.loc[partition].groupby(column)[column].agg('count')
        for value, count in group_counts.to_dict().items():
            p = count / total_count
            d = abs(p - global_freqs[value])
            if d_max is None or d > d_max:
                d_max = d
        return d_max

    def is_t_close(df, partition, sensitive_column, global_freqs, p=0.2):
        """
        :param               df: The dataframe for which to check l-diversity
        :param        partition: The partition of the dataframe on which to check l-diversity
        :param sensitive_column: The name of the sensitive column
        :param     global_freqs: The global frequencies of the sensitive attribute values
        :param                p: The maximum allowed Kolmogorov-Smirnov distance
        """
        if not sensitive_column in cate_col:
            raise ValueError("this method only works for categorical values")
        return t_closeness(df, partition, sensitive_column, global_freqs) <= p
        # --------------------------------함수 끝 --------------------
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
    with open("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".json", 'w') as f:
        json.dump(to_json, f)

    # df = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".csv")
    with open("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".json", 'r') as f:
        dtypes = json.load(f)
    columns = list(dtypes.keys())

    fff.to_csv("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + '.csv')

    rrf = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + '.csv', header=None, skiprows=1,
                      names=columns).astype(dtypes)

    file = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass

    # rrf = rrf.iloc[:,1:]
    # rrf.apply(pd.to_numeric, errors='coerce')
    spop = Synthpop()
    spop.fit(rrf, dtypes)
    synth_df = spop.generate(len(rrf))
    df2 = rrf.drop(first[0], axis=1)
    add_divide()
    synth_df2 = synth_df[first[0]]
    result = pd.concat([synth_df2, df2], axis=1)
    result.to_csv("/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv", index=False)


    aa = obj + '.csv'
    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv")
    # synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv")
    original_data = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))

    cate_col = []
    for i in range(0, len(synth_data.columns)):
        if synth_data.dtypes[i] == 'object':
            cate_col.append(synth_data.columns[i])

    cate_col2 = []
    for i in range(0, len(synth_data.columns)):
        if synth_data.dtypes[i] != 'object':
            cate_col2.append(synth_data.columns[i])

    full_spans = get_spans(original_data, original_data.index)

    feature_columns = cate_col2
    # sensitive_column = risk_category[0]
    sensitive_column = str(risk_category[0][0])

    origin_partitions = partition_dataset(original_data, feature_columns, sensitive_column, full_spans,
                                          is_k_anonymous)

    indexes = build_indexes(original_data)
    column_x, column_y = feature_columns[:2]
    rects = get_partition_rects(original_data, origin_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

    global_freqs = {}
    total_count = float(len(original_data))
    group_counts = original_data.groupby(sensitive_column)[sensitive_column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p

    # print(global_freqs)

    # 데이터 세트에 적용
    finished_t_close_partitions = partition_dataset(original_data, feature_columns, sensitive_column, full_spans,
                                                    lambda *args: is_k_anonymous(*args) and is_t_close(*args,
                                                                                                       global_freqs))
    # print(len(finished_t_close_partitions))

    dft = build_anonymized_dataset(original_data, finished_t_close_partitions, feature_columns, sensitive_column)

    # Let's see how t-closeness fares
    dft.sort_values([column_x, column_y, sensitive_column])
    # print(dft)
    dft.to_csv('/home/ubuntu/projects/FlaskProject/pybo/uploads/origin_dft' + obj + '.csv', index=False)

    # --------------------------------------- 위에가 원본 T근접성 구하기----------------

    full_spans = get_spans(synth_data, synth_data.index)

    synth_partitions = partition_dataset(synth_data, feature_columns, sensitive_column, full_spans, is_k_anonymous)

    indexes = build_indexes(synth_data)
    column_x, column_y = feature_columns[:2]
    rects = get_partition_rects(synth_data, synth_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

    global_freqs = {}
    total_count = float(len(synth_data))
    group_counts = synth_data.groupby(sensitive_column)[sensitive_column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p

    # print(global_freqs)

    # 데이터 세트에 적용
    finished_t_close_partitions = partition_dataset(synth_data, feature_columns, sensitive_column, full_spans,
                                                    lambda *args: is_k_anonymous(*args) and is_t_close(*args,
                                                                                                       global_freqs))
    # print(len(finished_t_close_partitions))

    dft = build_anonymized_dataset(synth_data, finished_t_close_partitions, feature_columns, sensitive_column)

    # Let's see how t-closeness fares
    dft.sort_values([column_x, column_y, sensitive_column])
    # print(dft)
    dft.to_csv('/home/ubuntu/projects/FlaskProject/pybo/uploads/synth_dft' + obj + '.csv', index=False)

    # --------------------- 여기까지 재현 T근접성---------------------------

    aa = risk_num[0][0]
    synth_data = synth_data[aa]
    original_data = original_data[aa]
    synth_data = synth_data.astype('float64')
    original_data = original_data.astype('float64')
    # print(synth_data)
    '''
    synth_data = synth_data.sort_values(aa)
    original_data = original_data.sort_values(aa)
    '''
    origin_identifier = np.array(original_data)
    synth_identifier = np.array(synth_data)

    vali = pd.read_csv('/home/ubuntu/projects/FlaskProject/pybo/uploads/origin_dft' + obj + '.csv')

    # col = ['age','count']
    col = [aa, 'count']
    # vali = vali[col]
    vali = vali[col]

    ori3 = []
    ori3 = list(vali[aa])
    ori33 = list(vali['count'])
    ori333 = []
    for i in range(0, len(ori3)):
        if ori33[i] > 1:
            for i in range(0, ori33[i]):
                ori333.append(ori3[i])
        else:
            ori333.append(ori3[i])

    vali_identifier = np.array(ori333)

    EMD_values = pyemd.emd_samples(
        first_array=origin_identifier,
        second_array=vali_identifier,
        extra_mass_penalty=0.0,
        distance='euclidean',
        normalized=True,
        bins='auto',
        range=None
    )
    # print('원본 : ' + str(EMD_values))

    vali2 = pd.read_csv('/home/ubuntu/projects/FlaskProject/pybo/uploads/synth_dft' + obj + '.csv')

    # col = ['age','count']
    col = [aa, 'count']
    # vali = vali[col]
    vali2 = vali2[col]
    syn3 = []
    syn3 = list(vali2[aa])
    syn33 = list(vali2['count'])
    syn333 = []
    for i in range(0, len(syn3)):
        if syn33[i] > 1:
            for i in range(0, syn33[i]):
                syn333.append(syn3[i])
        else:
            syn333.append(syn3[i])

    vali2_identifier = np.array(syn333)

    EMD_values2 = pyemd.emd_samples(
        first_array=synth_identifier,
        second_array=vali2_identifier,
        extra_mass_penalty=0.0,
        distance='euclidean',
        normalized=True,
        bins='auto',
        range=None
    )

    if(EMD_values > EMD_values2):
        df = fff.iloc[:10]
        result2 = result.iloc[:10]


        return render_template('partsynth_generate.html', tables=[df.to_html()], titles=[''],
                               tables2=[result2.to_html()],
                               titles2=[''], origin=EMD_values, synth=EMD_values2)
    else:
        return render_template('loading2.html')


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
        save_file = '/home/ubuntu/projects/FlaskProject/pybo/synth_dir/' + count2[j] + '.csv'
        bucket.download_file(obj_file, save_file)

    for k in range(0, len(count2)):
        syn = "/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + str(count2[0]) + ".csv"
        result = send_file(syn, as_attachment=True)
        return result

    for l in range(0, len(count2)):
        file = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + count2[l] + ".csv"
        try:
            os.remove(file)
        except OSError:
            pass

    return render_template('test.html')

    # return str(list1)


index_add_counter = []


def syn_down(filename):
    syn = "/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + filename + ".csv"
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

    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv")
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
        plt.savefig('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/' + obj + 'dis.png')

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

    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv")
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

    target_image = Image.open('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/baseimg.png')
    draw = ImageDraw.Draw(target_image)
    font = ImageFont.truetype("/home/ubuntu/projects/FlaskProject/pybo/static/arial.ttf", 21)
    draw.text((10, 10), list, fill="black", font=font)
    target_image.save('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/' + obj + 'originreg.png')
    target_image.close()

    target_image2 = Image.open('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/baseimg2.png')
    font2 = ImageFont.truetype("/home/ubuntu/projects/FlaskProject/pybo/static/arial.ttf", 21)
    draw2 = ImageDraw.Draw(target_image2)
    draw2.text((10, 10), list2, fill="black", font=font2)
    target_image2.save('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/' + obj + 'synthreg.png')
    target_image2.close()

    return render_template('regression.html', origin_file='/img_dir/' + obj + 'originreg.png',
                           synth_file='/img_dir/' + obj + 'synthreg.png')


# 상관관계 분석 - 수치형만
@bp.route('/correlation', methods=['GET', 'POST'])
def correlation():
    obj = g.user.username
    plt.close()
    # original_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".csv")
    s3 = boto3.resource('s3')
    bucket_name = 'origindir'
    bucket = s3.Bucket(name=bucket_name)
    aa = obj + '.csv'
    original_data = pd.read_csv(io.BytesIO(bucket.Object(aa).get()['Body'].read()))

    # original_data = original_data.iloc[:, 1:]
    synth_data = pd.read_csv("/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv")
    corr_df = original_data.corr()
    corr_df = corr_df.apply(lambda x: round(x, 2))

    ax = sns.heatmap(corr_df, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/' + obj + 'origincorr.png')
    plt.close()
    corr_df2 = synth_data.corr()
    corr_df2 = corr_df2.apply(lambda x: round(x, 2))

    ax2 = sns.heatmap(corr_df2, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('/home/ubuntu/projects/FlaskProject/pybo/static/img_dir/' + obj + 'synthcorr.png')
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

    local_file = "/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv"
    obj_file = obj + str(index_add_counter) + '.csv'
    bucket.upload_file(local_file, obj_file)

    file = "/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass
    stst = obj + str(index_add_counter)
    return render_template('syn_store.html', stst = stst)


@bp.route('/hello3')
def hello_pybo3():
    obj = g.user.username

    file = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".csv"
    try:
        os.remove(file)
    except OSError:
        pass

    syn = "/home/ubuntu/projects/FlaskProject/pybo/synth_dir/" + obj + str(index_add_counter) + ".csv"

    file2 = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + ".json"
    try:
        os.remove(file2)
    except OSError:
        pass

    file3 = "/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + "%.png"
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
            f.save("/home/ubuntu/projects/FlaskProject/pybo/uploads/" + obj + '.csv')
        else:
            return render_template('extension_error.html')


        return render_template('loading.html', my_data = some_data)


risk_category = []



risk_num = []




