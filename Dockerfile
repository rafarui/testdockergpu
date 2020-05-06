FROM python:3.6

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt

ARG GITTOKEN
ARG USER='totvslabs'

ARG REPO='pyCarol'
ARG BRANCH='dev-k8s-luigi'
RUN git clone -b$BRANCH  --depth 1 https://${GITTOKEN}@github.com/$USER/$REPO.git

WORKDIR /app/pyCarol
RUN pip install -r requirements.txt

WORKDIR /app

ADD . /app

CMD /app/run.sh
