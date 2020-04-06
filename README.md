Listwise Reformer (LWR)

## Steps to reproduce paper results

1. Clone this repo
```
    git clone REPO
```

2. Create virtual env, activate it and install requirements.txt
```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    cd list_wise_reformer
    pip install -e .`
```

3. Do either :
    - run ./download_data.sh and ./run_datasets_creation.sh or
    - download preprocessed data from URL

4. Run main.py from LWR
```
    TASK=ml25m
    REPO_DIR=/ssd/home/gustavo/recsys2020penha/
    python main.py \
        --seed 42 \
        --num_epochs 200 \
        --data_folder $REPO_DIR/data/recommendation/ \
        --output_dir $REPO_DIR/data/output_data/lwr \
        --task $TASK \
        --validate_epochs 10 \
        --max_seq_len 2048 \
        --train_batch_size 60 \
        --val_batch_size 60 \
        --input_representation "text"\
        --loss "PointwiseRMSE"
```

## Simple examples of using LWR

1. model_example.py
2. trainer_example.py
