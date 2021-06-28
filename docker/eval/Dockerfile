FROM python:3.6

WORKDIR .
COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache -r requirements.txt

# CHANGE generator dir and remove the skip flags for metrics you want to run
CMD ["python", "-u", "/main/evaluator/evaluator.py" \
    ,"--generator_dir", "/resources/dp-cgan-new/mnist/gen"\
    ,"--gen_data_size", "10000"\
    ,"--dataset", "mnist"\
    ,"--save_location", "/resources/dp-cgan-new/eval_results/mnist"\
    ,"--num_batches", "10"\
    ,"--skip_FID" \
    ,"--skip_classifier_acc"\
    ,"--skip_IS"\
    ]