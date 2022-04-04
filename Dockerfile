FROM ubuntu:20.04

WORKDIR /app
COPY . /app
RUN apt-get update &&  apt upgrade -y
RUN apt install python3 -y
RUN apt install build-essential python3-pip libffi-dev python3-dev python3-setuptools libssl-dev -y
RUN pip3 install -r requirements.txt
RUN pip3 install py-synthpop
ENV FLASK_APP=pybo
ENV FLASK_ENV=development
#RUN mv ~/app/py_synthpop-0.1.2.dist-info /usr/local/lib/python3.9/site-packages
#RUN mv ~/app/synthpop /usr/local/lib/python3.9/site-packages
#RUN mv py-synthpop synthpop
#RUN cd ~

EXPOSE 5000
CMD ["flask","run"]