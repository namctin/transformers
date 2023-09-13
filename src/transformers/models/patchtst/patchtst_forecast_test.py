from transformers import PatchTSTConfig, PatchTSTForForecasting
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import torch


class TestDataset(Dataset):
    def __init__(self, x, y, seq_len=10, pred_len=10, use_labels=False, is_pred=False):
        self.seq_len = seq_len
        self.x = x
        self.y = y
        self.is_pred = is_pred
        self.pred_len = pred_len
        self.use_labels = use_labels

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - 1
        r_end = s_end + self.pred_len

        seq_x = self.x[s_begin:s_end]
        seq_y = np.array(self.y[r_begin])

        if self.is_pred:
            seq_y = self.x[s_end:r_end]
        elif self.use_labels:
            return {'past_values': seq_x, 'labels': seq_y}
        return {'past_values': seq_x, 'future_values': seq_y}

    def __len__(self):
        if self.is_pred:
            return len(self.x) - self.seq_len - self.pred_len + 1
        return len(self.x) - self.seq_len + 1


if __name__ == "__main__":

    n_classes = 3
    bs = 400
    n_features = 20
    pred_len = 7
    target_dim = 2
    x = torch.randn(bs, n_features)
    y = torch.randint(low=0, high=n_classes, size=(bs, 1))[:, 0]

    valid_asset_ds = train_asset_ds = TestDataset(x, y, seq_len=10, pred_len=pred_len, use_labels=False, is_pred=True)
    config = PatchTSTConfig(
        num_input_channels=n_features,
        num_classes=n_classes,
        context_length=10,
        patch_length=5,
        stride=5,
        batch_size=50,
        standardscale=None,
        context_points=10,
        encoder_layers=12,
        encoder_attention_heads=8,
        d_model=256,
        encoder_ffn_dim=1024,
        dropout=0.2,
        fc_dropout=0,
        r=0.4,
        prediction_length=pred_len,
        pooling=None
    )

    model = PatchTSTForForecasting(config)

    training_args = TrainingArguments(
        output_dir='./save_model/',
        num_train_epochs=5,
        per_device_train_batch_size=50,
        per_device_eval_batch_size=50,
        remove_unused_columns=False,
        use_cpu=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        label_names=['future_values'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_asset_ds,
        eval_dataset=valid_asset_ds,
    )
    trainer.train()
    print(model)
