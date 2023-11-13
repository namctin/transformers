from transformers import PatchTSMixerConfig, Trainer, TrainingArguments
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
import os
import numpy as np
import evaluate
from statsmodels.tsa.seasonal import seasonal_decompose


def stl_decompose(series, seasonal_period):
    decomposed = seasonal_decompose(
        # series, model="additive", period=seasonal_period, extrapolate_trend="freq"
        series,
        model="additive",
        extrapolate_trend="freq",
        period=seasonal_period,
    )
    return (
        np.array(decomposed.seasonal),
        np.array(decomposed.trend),
        np.array(decomposed.resid),
    )


def stl_decomposition_array(data, seasonal_period=24, loess_span=0.15):
    t, c = data.shape
    result = np.zeros((3 * t, c))

    for j in range(c):
        # print("j",j)
        series = data[:, j]
        seasonal, trend, resid = stl_decompose(series, seasonal_period)
        result[0:t, j] = seasonal
        result[t : 2 * t, j] = trend
        result[2 * t : 3 * t, j] = resid

        # decomposed = seasonal_decompose(series, period=seasonal_period, model='additive', filt=np.array([loess_span, 0]))
        # result[0:t, j] = decomposed.seasonal
        # result[t:2*t, j] = decomposed.trend
        # result[2*t:3*t, j] = decomposed.resid

    return result


from sklearn.preprocessing import StandardScaler
import pandas as pd


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path="/dccstor/dnn_forecasting/FM/data/ETDataset/ETT-small/",
        split="train",
        size=None,
        features="M",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        use_time_features=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x = stl_decomposition_array(seq_x, seasonal_period=96)
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #         if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        #         else: return _torch(seq_x, seq_y)
        return {
            "past_values": torch.Tensor(seq_x),
            "target_values": torch.Tensor(seq_y),
        }

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class ETTDataset(Dataset):
    def __init__(
        self,
        root_path="/dccstor/dnn_forecasting/FM/data/ETDataset/ETT-small/",
        data_file="ETTh1.csv",
        seq_len=128,
        pred_len=32,
        split="train",
        scale=True,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        # init
        assert split in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[split]

        self.scale = scale

        self.root_path = root_path
        self.data_file = data_file
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_file))
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        df_raw["month"] = df_raw["date"].dt.month
        df_raw["day"] = df_raw["date"].dt.day
        df_raw["weekday"] = df_raw["date"].dt.weekday
        df_raw["hour"] = df_raw["date"].dt.hour
        print(df_raw.columns)
        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x = stl_decomposition_array(seq_x)
        return {
            "past_values": torch.Tensor(seq_x),
            "target_values": torch.Tensor(seq_y),
        }

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


FORECAST_LEN = 96
n_features = 7 + 4
SEQ_LEN = 512
seq_len = SEQ_LEN
patch_len = 16
stride = patch_len


dset_train = ETTDataset(split="train", seq_len=SEQ_LEN, pred_len=FORECAST_LEN)
dset_val = ETTDataset(split="val", seq_len=SEQ_LEN, pred_len=FORECAST_LEN)
dset_test = ETTDataset(split="test", seq_len=SEQ_LEN, pred_len=FORECAST_LEN)


dd = dset_val.__getitem__(0)
print(dd["past_values"].shape, dd["target_values"].shape)


num_patches = seq_len // patch_len
print(num_patches)


from transformers import PatchTSMixerForForecasting

k = 3

forecast_config = PatchTSMixerConfig(
    num_input_channels=n_features,
    seq_len=SEQ_LEN * k,
    patch_len=patch_len,
    stride=stride,
    num_features=32,
    num_layers=4,
    dropout=0.7,
    mode="common_channel",
    revin=True,
    expansion_factor=3,
    head_dropout=0.7,
    forecast_len=FORECAST_LEN * k,
    scaling=False,
    decompose_k=k,
    seed_number=40,
)

forecast_model = PatchTSMixerForForecasting(forecast_config)

forecast_args = TrainingArguments(
    output_dir="/dccstor/tsfm-irl/vijaye12/workdir/temp/decompose/dump/etth1/direct_forecast/checkpoint",
    overwrite_output_dir=True,
    learning_rate=0.0001,
    num_train_epochs=100,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=1024,
    report_to="tensorboard",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    logging_dir="/dccstor/tsfm-irl/vijaye12/workdir/temp/decompose/dump/etth1/direct_forecast/logs",  # Make sure to specify a logging directory
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
    dataloader_num_workers=16,
)


from transformers import EarlyStoppingCallback

# Create the early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.01,  # Minimum improvement required to consider as improvement
)


def compute_metrics(eval_pred):
    # print(eval_pred.prediction_logits.shape)
    # metric = evaluate.load("mse")
    pred = eval_pred[0][0]
    target = eval_pred[0][1]
    o = np.mean((target - pred) ** 2)
    # metric.compute(predictions=pred, references=target)
    return {"mse": o}
    # print(pred.shape, target.shape)
    # return np.mean((target - pred) ** 2)


forecast_trainer = Trainer(
    model=forecast_model,
    args=forecast_args,
    train_dataset=dset_train,
    eval_dataset=dset_val,
    callbacks=[early_stopping_callback],
    compute_metrics=compute_metrics,
)


forecast_trainer.train()

print("evaluation")
print(forecast_trainer.evaluate(dset_test))
