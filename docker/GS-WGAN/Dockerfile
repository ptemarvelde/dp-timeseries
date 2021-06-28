FROM python:3.6

WORKDIR .
COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache -r requirements.txt

#ENV KAGGLE_USERNAME=
#ENV KAGGLE_KEY=


CMD ["python", "-u", "/main/source/main.py" \
    ,"-data", "ptb"\
    ,"-name", "ptb1"\
#    ,"-ldir", "/main/results/ptb/pretrain/ptb_pretrain_1"\
    ,"-vstep", "1000"\
    ,"-sstep", "1000"\
    ,"-ndis", "100"\
    ,"-iters", "30000"]