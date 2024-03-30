FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libcairo2 build-essential libbz2-dev libdb-dev libreadline-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libsqlite3-dev libssl-dev zlib1g-dev uuid-dev python3 python3-dev curl python3-pip
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3

RUN pip install tensorflow-gpu==2.10.0
RUN pip install tensorflow==2.10.0
RUN pip install matplotlib
RUN pip install japanize_matplotlib
RUN pip install jupyter
RUN pip install protein-bert
RUN pip install numpy==1.21.0
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install biopython
RUN pip install torch
RUN pip install transformers

WORKDIR /home
EXPOSE 5004
ENTRYPOINT ["jupyter","notebook","--port=5004","--no-browser","--allow-root","--ip=0.0.0.0","--NotebookApp.token='kanami-yagimoto'"]
