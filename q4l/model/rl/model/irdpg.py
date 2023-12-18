from . import PortfolioActorModel, Reward


class iRDPGModel(PortfolioActorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def actual_forward(self, **kwargs):
        return super().actual_forward(**kwargs)


class iRDPGReward(Reward):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        return super().__call__(**kwargs)
