import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCorrelationConvBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        dilation_rates,
        dropout_rate,
    ):
        super(TemporalCorrelationConvBlock, self).__init__()

        # Create a sequence of dilated causal convolutions
        self.layers = nn.ModuleList()
        for dilation_rate in dilation_rates:
            self.layers.append(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    padding=dilation_rate * (kernel_size - 1) // 2,
                    dilation=dilation_rate,
                )
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            input_channels = (
                output_channels  # Update input channels for the next layer
            )

        # Final 1x1 convolution layer
        self.conv1x1 = nn.Conv1d(output_channels, output_channels, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.conv1x1(x)
        return x


class PPNModel(nn.Module):
    def __init__(
        self, num_assets, num_factors, hidden_size, portfolio_vector_size
    ):
        super(PPNModel, self).__init__()

        # Sequential Information Net
        self.lstm = nn.LSTM(
            input_size=num_factors, hidden_size=hidden_size, batch_first=True
        )

        # Correlation Information Net
        # Assuming kernel_size and dropout_rate are given or can be determined
        kernel_size = 3
        dropout_rate = 0.1
        dilation_rates = [1, 2, 4]  # Example dilation rates for the TCCBs
        self.tccb1 = TemporalCorrelationConvBlock(
            num_factors, 32, kernel_size, dilation_rates, dropout_rate
        )
        self.tccb2 = TemporalCorrelationConvBlock(
            32, 64, kernel_size, dilation_rates, dropout_rate
        )
        self.tccb3 = TemporalCorrelationConvBlock(
            64, 128, kernel_size, dilation_rates, dropout_rate
        )
        self.conv4 = nn.Conv1d(
            128, 128, kernel_size=3, padding=1
        )  # Padding to keep the size the same

        # Decision-making Module
        self.decision_fc = nn.Linear(hidden_size + 128, portfolio_vector_size)

    def forward(self, x, last_portfolio_vector, cash_bias):
        # Sequential Information Net
        x_seq, (hn, cn) = self.lstm(x)
        x_seq = hn[-1]  # we take the output of the last LSTM cell

        # Correlation Information Net
        x_corr = self.tccb1(x)
        x_corr = self.tccb2(x_corr)
        x_corr = self.tccb3(x_corr)
        x_corr = self.conv4(x_corr)
        x_corr = torch.mean(x_corr, dim=2)  # global average pooling

        # Combine features from both nets
        combined_features = torch.cat([x_seq, x_corr], dim=1)

        # Decision-making Module
        combined_features = torch.cat(
            [combined_features, last_portfolio_vector, cash_bias.unsqueeze(1)],
            dim=1,
        )
        portfolio_vector = self.decision_fc(combined_features)
        portfolio_vector = F.softmax(portfolio_vector, dim=1)

        return portfolio_vector


class PortfolioReward:
    """This reward computes a risk-sensitive reward based on returns and a
    transaction cost constraint.

    It is based on the formula provided in the image, which includes a term for
    average excess returns, a penalty for variance of excess returns, and a
    transaction cost term.

    """

    def __init__(self, lambda_risk: float, gamma_transaction: float):
        """Initialize the PortfolioReward.

        Parameters
        ----------
        lambda_risk : float
            The risk aversion parameter that scales the variance penalty term.
        gamma_transaction : float
            The transaction cost scale parameter.

        """
        super().__init__()
        self.lambda_risk = lambda_risk
        self.gamma_transaction = gamma_transaction
        self.reset()

    def reset(self):
        """Reset the reward computation for a new episode."""
        self.new = True
        self.returns = []  # List to store returns
        self.actions = (
            []
        )  # List to store actions for transaction cost computation

    def __call__(self, state, action) -> float:
        """Compute the reward based on the current environment snapshot.

        Parameters
        ----------
        env_snapshot : Dict
            A dictionary containing 'asset', 'returns', and 'actions' for the current timestep.

        Returns
        -------
        float
            The computed reward for the current timestep.

        """
        # Unpack the environment snapshot
        close = state[:-4]
        if self.new:
            ret = torch.zeros_like(close)
            self.prev_close = close
            self.new = False
        else:
            ret = (close / self.prev_close - 1.0) * action

        # Update the returns and actions lists
        self.returns.append(ret)
        if len(self.actions) > 0:
            self.actions.append(action)

        # Compute risk-sensitive part of the reward
        if len(self.returns) > 1:
            avg_excess_return = torch.mean(torch.stack(self.returns))
            excess_return_var = torch.var(torch.stack(self.returns))
            risk_sensitive_reward = (
                avg_excess_return - self.lambda_risk * excess_return_var
            )
        else:
            risk_sensitive_reward = 0.0

        # Compute transaction cost part of the reward
        if len(self.actions) > 1:
            transaction_costs = sum(
                F.l1_loss(act, self.actions[i - 1])
                for i, act in enumerate(self.actions[1:], start=1)
            )
            transaction_cost_penalty = (
                self.gamma_transaction
                * transaction_costs
                / (len(self.actions) - 1)
            )
        else:
            transaction_cost_penalty = 0.0

        # The final reward is the combination of the risk-sensitive reward and the transaction cost penalty
        reward = risk_sensitive_reward - transaction_cost_penalty
        return reward.item()  # Return the reward as a Python float
