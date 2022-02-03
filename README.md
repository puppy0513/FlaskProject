# 7조 FlaskProject

먼저 폴더 아무거나 만드시고 여기 안의 파일들 넣어주세요(git clone)

그다음 cmd창에서 그 폴더에 들어가시고

set flask_app=pybo
set flask_env=development
설정해준 뒤

flask run 해주시면 됩니다

![image](https://user-images.githubusercontent.com/92296556/151277485-3b4535cd-538c-4cef-a0c9-464b3c3d7d67.png

필요한 라이브러리 다운로드 받으시고 사용하시기 바랍니다

# 20220127 추가사항 V2.0
- 파일업로드 및 업로드 시 용량제한(현재 16MB)
- csv 파일 지정 경로 저장
# 20220128 추가사항 V3.0
- 파일업로드 시 현재 로그인한 아이디로 id.csv파일 생성
- 데이터 타입을 식별 후 로그인한 id.json 파일 생성
- 재현데이터 생성 후 id.csv파일로 synth_dir로 저장
# V3.2
- 업로드 된 파일 분포도 시각화 이미지 static/img_dir에 저장(id + dis.png로 저장)
- distribution.html 에서 출력

# 20220203 추가사항 V4.0
- 수치형 데이터 상관관계 분석 및 이미지 출력(static/img_dir에 저장)
- 생성된 재현데이터 다운로드 가능(userid.csv로)
