#!/bin/bash


cd /home/ubuntu/projects
export FLASK_APP=pybo
export FLASK_ENV=development
cd /home/ubuntu/venvs/myproject/bin
. activate

cd /home/ubuntu/projects/myproject
. flask db upgrade
flask run --host=0.0.0.0