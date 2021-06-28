FROM python:3.6

RUN python -m pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache -r requirements.txt

CMD ["python3", "-u", "/main/dpcgan/dp_cgan_reference_im.py"]

