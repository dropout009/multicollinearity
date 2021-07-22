from __future__ import annotations
from typing import Any, Callable
from dataclasses import dataclass, asdict
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


@dataclass
class Simulator:
    n_simulations: int
    n_samples: int
    n_features: int
    rho: float
    sigma_u: float

    def generate_simulation_data(
        self,
        n_simulations: int,
        n_samples: int,
        n_features: int,
        rho: float,
        sigma_u: float,
    ) -> tuple(np.ndarray, np.ndarray):
        """シミュレーションデータの生成"""

        # シミュレーションの設定
        # X0とX1のみ相関がありうる設定にする
        mean = np.zeros(n_features)
        cov = np.eye(n_features)
        cov[0, 1] = cov[1, 0] = rho

        # 特徴量とノイズを生成
        X = np.random.multivariate_normal(mean, cov, size=(n_simulations, n_samples))
        u = np.random.normal(0, sigma_u, size=(n_simulations, n_samples))

        # 係数はすべて1なので、単に和をとる
        y = X.sum(axis=2) + u

        return (X, y)

    @staticmethod
    def estimate(
        X: np.ndarray,
        y: np.ndarray,
        l2_lambda: float,
        n_simulations: int,
        n_samples: int,
        n_features: int,
        **kwargs,
    ) -> np.ndarray:
        """回帰係数を推定する"""

        XT = np.transpose(X, axes=(0, 2, 1))
        Y = y.reshape((-1, n_samples, 1))

        I = np.row_stack([np.eye(n_features)] * n_simulations)
        I = I.reshape(n_simulations, n_features, n_features)

        hbeta = np.linalg.inv(XT @ X + l2_lambda * I) @ (XT @ Y)

        return hbeta.reshape(-1, n_features)

    @staticmethod
    def calc_mean_var(
        hbeta: np.ndarray,
        target_key: str,
        target_values: list,
        n_features: int,
        **kwargs,
    ) -> pd.DataFrame:
        """シミュレーションでの回帰係数の平均と分散を計算する"""

        df = pd.DataFrame({target_key: target_values})
        df[[f"平均{j}" for j in range(n_features)]] = hbeta.mean(axis=1)
        df[[f"分散{j}" for j in range(n_features)]] = hbeta.var(axis=1)

        return df

    def simulate(
        self, target_key: str, target_values: list, l2_lambda: float = 0
    ) -> pd.DataFrame:
        """シミュレーションを実行する"""

        hbeta = np.zeros((len(target_values), self.n_simulations, self.n_features))
        params = asdict(self)
        for i, target_value in enumerate(target_values):
            params.update({target_key: target_value})
            X, y = self.generate_simulation_data(**params)
            hbeta[i, :, :] = self.estimate(X, y, l2_lambda, **params)

        return self.calc_mean_var(hbeta, target_key, target_values, **params)

    def plot(self, df: pd.DataFrame, target_key: str, kind: str) -> None:
        """シミュレーション結果を可視化"""
        
        fig, ax = plt.subplots()

        for j in range(self.n_features):
            ax.plot(df[target_key], df[f"{kind}{j}"], label=f"beta{j}")
        ax.legend()
        ax.set(xlabel=target_key, ylabel=kind)
        fig.suptitle(f"{target_key}と回帰係数の{kind}の関係")
        
        fig.savefig(f"output/{target_key}と回帰係数の{kind}の関係")
        fig.show()
        
        