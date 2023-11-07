
import json
import pandas as pd


def read_mmdet_train_log_json(file_path: str):

    df_train = []
    df_val = []

    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if 'lr' in record:
                df_train.append(record)
            else:
                df_val.append(record)

    df_train = pd.DataFrame(df_train)
    df_val = pd.DataFrame(df_val)

    if len(df_train):
        df_train = df_train.groupby('epoch').mean()
        print(df_train.columns.to_list())

    if len(df_val):
        df_val = df_val.groupby('step').mean()
        print(df_val.columns.to_list())
        print(f"最大map: ({df_val['coco/bbox_mAP'].idxmax()}){df_val['coco/bbox_mAP'].max()}")  # noqa

    return df_train, df_val


def read_ultrlytics_train_log_csv(file_path: str):

    re = pd.read_csv(file_path)
    columns = [i.strip() for i in re.columns.tolist()]
    re.columns = columns
    print(re.columns.to_list())

    return re
