import pandas as pd

from ...data.loader import Q4LDataLoader
from ...qlib.backtest.exchange import Exchange as QlibExchange
from ...utils.log import get_logger


class Q4LExchange(QlibExchange):
    def __init__(
        self,
        storage_backend: str,
        compute_backend: str,
        loader: Q4LDataLoader,
        pool: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loader = loader
        self.logger = get_logger(self)
        self.compute_backend_name = compute_backend
        self.pool = pool
        self.codes = self.loader.storage_backends[
            storage_backend
        ].get_ticker_list(pool=self.pool)
        self.rebuild_quote_df()
        self.quote = self.quote_cls(self.quote_df, freq=self.freq)

    def rebuild_quote_df(self):
        self.logger.info("Rebuilding quote_df...")
        all_fields = [f"{{disk:{x[1:]}}}" for x in self.all_fields]
        self.logger.info(
            f"all_fields: {self.all_fields}, No = {len(self.all_fields)}"
        )
        self.logger.info(f"number of codes = {len(self.codes)}")

        self.quote_df = self.loader.load_group_df(
            instruments=self.codes,
            expressions=all_fields,
            compute_backend=self.compute_backend_name,
            start_time=self.start_time,
            end_time=self.end_time,
            names=self.all_fields,
        )
        # self.quote_df.columns = all_fields
        self.logger.info(
            "Rebuilding quote_df done. Handle to qlib legacy code."
        )

        # check buy_price data and sell_price data
        for attr in ("buy_price", "sell_price"):
            pstr = getattr(self, attr)  # price string
            if self.quote_df[pstr].isna().any():
                self.logger.warning("{} field data contains nan.".format(pstr))

        # update trade_w_adj_price
        if (
            self.quote_df["$factor"].isna() & ~self.quote_df["$close"].isna()
        ).any():
            # The 'factor.day.bin' file not exists, and `factor` field contains `nan`
            # Use adjusted price
            self.trade_w_adj_price = True
            self.logger.warning(
                "factor.day.bin file not exists or factor contains `nan`. Order using adjusted_price."
            )
            if self.trade_unit is not None:
                self.logger.warning(
                    f"trade unit {self.trade_unit} is not supported in adjusted_price mode."
                )
        else:
            # The `factor.day.bin` file exists and all data `close` and `factor` are not `nan`
            # Use normal price
            self.trade_w_adj_price = False
        # update limit
        self._update_limit(self.limit_threshold)

        # concat extra_quote
        if self.extra_quote is not None:
            # process extra_quote
            if "$close" not in self.extra_quote:
                raise ValueError("$close is necessray in extra_quote")
            for attr in "buy_price", "sell_price":
                pstr = getattr(self, attr)  # price string
                if pstr not in self.extra_quote.columns:
                    self.extra_quote[pstr] = self.extra_quote["$close"]
                    self.logger.warning(
                        f"No {pstr} set for extra_quote. Use $close as {pstr}."
                    )
            if "$factor" not in self.extra_quote.columns:
                self.extra_quote["$factor"] = 1.0
                self.logger.warning(
                    "No $factor set for extra_quote. Use 1.0 as $factor."
                )
            if "limit_sell" not in self.extra_quote.columns:
                self.extra_quote["limit_sell"] = False
                self.logger.warning(
                    "No limit_sell set for extra_quote. All stock will be able to be sold."
                )
            if "limit_buy" not in self.extra_quote.columns:
                self.extra_quote["limit_buy"] = False
                self.logger.warning(
                    "No limit_buy set for extra_quote. All stock will be able to be bought."
                )
            assert set(self.extra_quote.columns) == set(
                self.quote_df.columns
            ) - {"$change"}
            self.quote_df = pd.concat(
                [self.quote_df, self.extra_quote], sort=False, axis=0
            )
