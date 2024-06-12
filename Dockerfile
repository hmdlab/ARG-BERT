FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG port
ARG password

ENV PORT=$port
ENV PASSWORD=$password

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libcairo2 build-essential libbz2-dev libdb-dev libreadline-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libsqlite3-dev libssl-dev zlib1g-dev uuid-dev python3 python3-dev curl python3-pip
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3

RUN pip install tensorflow-gpu==2.10.0
RUN pip install tensorflow==2.10.0
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install numpy==1.21.0
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install biopython

WORKDIR /home

# RUN mkdir -p /root/.jupyter
# COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

EXPOSE $port
ENTRYPOINT ["sh", "-c","jupyter notebook --port=$PORT --no-browser --allow-root --ip=0.0.0.0 --NotebookApp.token=$PASSWORD"]
