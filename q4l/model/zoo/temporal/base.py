import typing as tp

from lightning.pytorch import LightningModule


class BaseTemporalEncoder(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze(0)
        emb = self.model(x)  # (B, S, D)
        # if self.keep_seq:
        #     emb = emb
        # else:
        #     emb = emb[:, -1, :]
        # if torch.any(torch.isnan(emb)):
        #     torch.nan_to_num(emb, nan=0.0)
        return {"emb": emb}
