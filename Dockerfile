FROM python:3.7

# set work directory
WORKDIR /home/project

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .