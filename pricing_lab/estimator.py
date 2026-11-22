"""替代与净增需求估计。

模型：可用性条件 logit（availability-aware conditional logit）。
- 效用 U(套餐) = 套餐截距 + 价格系数 × 价格，外部选项（不买）效用为 0；
- 选择集严格取旅客在随机轮换触达中实际看到的选项；没看到的套餐不进入选择集，
  因此「未看到」不会被误判为「看到后拒绝」；
- 未付款旅程的实际选择记为外部选项（净增需求由此识别：降价使外部选项转化）；
- 自选旧价旅程与非随机时段旅程不参与拟合，只做适用性诊断。

随机轮换单位是时段内整批客群，因此：
- 不确定区间用「时段整批重抽样」的 cluster bootstrap，不拆散同一时段的旅程；
- 历史留出（holdout）也按时段整批划分折外集，折外指标反映同一轮换单位。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .evidence import JourneyEvidence, build_evidence, causal_rows
from .io import DataBundle
from .models import PLANS

_PLAN_INDEX = {p: i for i, p in enumerate(PLANS)}
_NO_BUY = len(PLANS)
# 价格以百元为单位参与优化，避免百元级量纲导致的步长失控；
# 对外的 price_coef 仍按每元解释。
PRICE_SCALE = 100.0


@dataclass(frozen=True)
class ChoiceRow:
    """拟合用单行：选择集内价格（外部选项占最后一位，价格为 0）。"""

    journey: str
    period: str
    segment: str
    available: tuple[bool, bool, bool, True]
    prices: tuple[float, float, float, 0.0]
    choice: int  # 0..2 套餐，3 = 外部选项


@dataclass
class DemandModel:
    """拟合后的条件 logit 参数与重抽样分布。"""

    intercepts: dict[str, float]
    price_coef: float
    segments: tuple[str, ...]
    periods: tuple[str, ...]
    rows: tuple[ChoiceRow, ...]
    converged: bool
    message: str
    bootstrap_params: np.ndarray | None = None
    cl_level: float = 0.9

    # ---- 概率 ----
    def choice_probabilities(
        self, available: dict[str, bool], prices: dict[str, float]
    ) -> dict[str, float]:
        """给定选择集与价格，返回各套餐与外部选项 no_buy 的概率。"""
        utils = np.zeros(len(PLANS) + 1)
        for i, plan in enumerate(PLANS):
            if available.get(plan, False):
                utils[i] = self.intercepts[plan] + self.price_coef * prices[plan]
        utils[_NO_BUY] = 0.0
        chosen_utils = np.array(
            [
                utils[i]
                for i in range(len(PLANS) + 1)
                if i == _NO_BUY or available.get(PLANS[i], False)
            ]
        )
        chosen_utils -= chosen_utils.max()
        exp_u = np.exp(chosen_utils)
        probs_flat = exp_u / exp_u.sum()
        out: dict[str, float] = {}
        k = 0
        for i, plan in enumerate(PLANS):
            if available.get(plan, False):
                out[plan] = float(probs_flat[k])
                k += 1
        out["no_buy"] = float(probs_flat[k])
        return out

    def expected_demand(
        self,
        cohort: list[JourneyEvidence],
        available: dict[str, bool],
        prices: dict[str, float],
    ) -> dict[str, float]:
        """对一批客群按各自选择集计算期望套餐需求与未转化数。"""
        totals = {p: 0.0 for p in PLANS}
        totals["no_buy"] = 0.0
        for ev in cohort:
            row_available = {p: ev.plan_evidence[p] == "seen" or available.get(p, False)
                             for p in PLANS}
            probs = self.choice_probabilities(row_available, prices)
            for key, val in probs.items():
                totals[key] += val
        return totals

    # ---- 不确定区间 ----
    def demand_distribution(
        self,
        cohort: list[JourneyEvidence],
        available: dict[str, bool],
        prices: dict[str, float],
        n_draws: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """用 bootstrap 参数分布模拟期望需求的抽样分布（时段整批口径）。"""
        rng = rng or np.random.default_rng()
        if self.bootstrap_params is None:
            raise ValueError("模型未带 bootstrap 分布，无法给出不确定区间")
        draws = self.bootstrap_params[
            rng.integers(0, len(self.bootstrap_params), size=n_draws)
        ]
        results = {p: np.zeros(n_draws) for p in PLANS}
        results["no_buy"] = np.zeros(n_draws)
        for d, params in enumerate(draws):
            model = _model_with_params(self, params)
            demand = model.expected_demand(cohort, available, prices)
            for key in results:
                results[key][d] = demand[key]
        return results

    def choice_simulation(
        self,
        cohort: list[JourneyEvidence],
        available: dict[str, bool],
        prices: dict[str, float],
        n_draws: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """联合模拟：参数抽样与每个旅程的离散选择共用同一批随机数。

        每次抽取先从 bootstrap 分布取一组模型参数，再让客群中每个旅程
        按其选择集概率独立抽取实际选择，得到整数需求。
        容量与价差检验必须共用本方法的同一次抽取，禁止分别挑最有利预测。
        """
        rng = rng or np.random.default_rng()
        if self.bootstrap_params is None:
            raise ValueError("模型未带 bootstrap 分布，无法给出不确定区间")
        n = len(cohort)
        param_idx = rng.integers(0, len(self.bootstrap_params), size=n_draws)
        params = self.bootstrap_params[param_idx]  # n_draws × 4
        intercept = params[:, : len(PLANS)]        # n_draws × 3
        beta = params[:, -1][:, None]              # n_draws × 1
        keys = list(PLANS) + ["no_buy"]
        counts = {p: np.zeros(n_draws) for p in PLANS}
        counts["no_buy"] = np.zeros(n_draws)
        for ev in cohort:
            avail_idx = [
                i for i, p in enumerate(PLANS)
                if ev.plan_evidence[p] == "seen" or available.get(p, False)
            ]
            price_vec = np.array([prices[PLANS[i]] for i in avail_idx])
            # n_draws × 可用选项（含外部选项，效用 0）
            utils = intercept[:, avail_idx] + beta * price_vec[None, :]
            utils = np.column_stack([utils, np.zeros(n_draws)])
            utils -= utils.max(axis=1, keepdims=True)
            exp_u = np.exp(utils)
            probs = exp_u / exp_u.sum(axis=1, keepdims=True)
            draws = rng.random(n_draws)
            chosen = (draws[:, None] < np.cumsum(probs, axis=1)).argmax(axis=1)
            chosen_keys = avail_idx + [len(PLANS)]
            for d, k_idx in enumerate(chosen):
                counts[keys[chosen_keys[k_idx]]][d] += 1
        return counts

    def interval(
        self, samples: np.ndarray, cl_level: float | None = None
    ) -> tuple[float, float]:
        level = self.cl_level if cl_level is None else cl_level
        alpha = (1 - level) / 2
        return float(np.quantile(samples, alpha)), float(
            np.quantile(samples, 1 - alpha)
        )


@dataclass
class DemandEstimate:
    """一次策略试算的需求结论（点估 + 区间 + 替代/净增分解）。"""

    policy_version: str
    cohort_size: int
    baseline_demand: dict[str, float]
    policy_demand: dict[str, float]
    demand_intervals: dict[str, tuple[float, float]]
    substitution: dict[str, dict[str, float]]
    net_increment: float
    net_increment_interval: tuple[float, float]
    support_gaps: list[str] = field(default_factory=list)


def build_choice_rows(bundle: DataBundle) -> tuple[ChoiceRow, ...]:
    """从因果可比旅程构造选择行；自选/非随机旅程被排除。"""
    rows: list[ChoiceRow] = []
    for ev in causal_rows(build_evidence(bundle)):
        prices: dict[str, float] = {}
        available = {p: False for p in PLANS}
        for plan, seen in ev.plan_evidence.items():
            if seen == "seen":
                available[plan] = True
                prices[plan] = ev.seen_prices[plan][-1]
        choice = _PLAN_INDEX[ev.chosen_plan] if ev.chosen_plan else _NO_BUY
        rows.append(
            ChoiceRow(
                journey=ev.journey,
                period=ev.period,
                segment=ev.segment,
                available=tuple(available[p] for p in PLANS) + (True,),
                prices=tuple(prices.get(p, 0.0) for p in PLANS) + (0.0,),
                choice=choice,
            )
        )
    return tuple(rows)


def _design_arrays(
    rows: tuple[ChoiceRow, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把选择行展开为向量化设计矩阵（n 行 × 4 个选项）。"""
    n = len(rows)
    avail = np.zeros((n, len(PLANS) + 1), dtype=bool)
    prices = np.zeros((n, len(PLANS) + 1))
    choice = np.zeros(n, dtype=int)
    for i, row in enumerate(rows):
        avail[i] = np.array(row.available, dtype=bool)
        prices[i] = np.array(row.prices)
        choice[i] = row.choice
    return avail, prices, choice


def _loss_and_grad(
    params: np.ndarray,
    avail: np.ndarray,
    prices: np.ndarray,
    choice: np.ndarray,
) -> tuple[float, np.ndarray]:
    """条件 logit 的负对数似然与解析梯度（向量化）。"""
    intercepts = params[: len(PLANS)]
    beta_scaled = params[-1]
    scaled_prices = prices / PRICE_SCALE
    utils = np.zeros_like(prices, dtype=float)
    utils[:, : len(PLANS)] = avail[:, : len(PLANS)] * (
        intercepts + beta_scaled * scaled_prices[:, : len(PLANS)]
    )
    # 数值稳定的 softmax，仅对可用选项；平移常数只用于稳定，不进似然值
    masked = np.where(avail, utils, -np.inf)
    shift = masked.max(axis=1, keepdims=True)
    masked = masked - shift
    exp_u = np.where(avail, np.exp(masked), 0.0)
    probs = exp_u / exp_u.sum(axis=1, keepdims=True)
    log_denom = shift[:, 0] + np.log(exp_u.sum(axis=1))
    chosen_util = utils[np.arange(len(choice)), choice]
    nll = float((log_denom - chosen_util).sum())

    grad = np.zeros_like(params)
    indicator = np.zeros_like(probs)
    indicator[np.arange(len(choice)), choice] = 1.0
    residual = probs - indicator  # n × 4，最后一列为外部选项（不参数化，不进梯度）
    for j in range(len(PLANS)):
        grad[j] = (residual[:, j] * avail[:, j]).sum()
    grad[-1] = (
        residual[:, : len(PLANS)]
        * avail[:, : len(PLANS)]
        * scaled_prices[:, : len(PLANS)]
    ).sum()
    return nll, grad


def _fit_once(
    rows: tuple[ChoiceRow, ...], x0: np.ndarray | None = None
) -> tuple[np.ndarray, bool, str]:
    avail, prices, choice = _design_arrays(rows)
    # 优化变量中的价格系数按百元计，初值/热启动按每元传入并换算
    if x0 is None:
        x0_scaled = np.zeros(len(PLANS) + 1)
    else:
        x0_scaled = np.concatenate([x0[: len(PLANS)], [x0[-1] * PRICE_SCALE]])
    result = minimize(
        _loss_and_grad, x0_scaled, args=(avail, prices, choice), jac=True,
        method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12},
    )
    params = result.x.copy()
    params[-1] = params[-1] / PRICE_SCALE  # 换算回每元口径
    return params, bool(result.success), str(result.message)


def _model_with_params(base: DemandModel, params: np.ndarray) -> DemandModel:
    return DemandModel(
        intercepts={p: float(params[i]) for i, p in enumerate(PLANS)},
        price_coef=float(params[-1]),
        segments=base.segments,
        periods=base.periods,
        rows=base.rows,
        converged=True,
        message="bootstrap draw",
        cl_level=base.cl_level,
    )


def fit_demand(
    bundle: DataBundle,
    n_bootstrap: int = 1000,
    cl_level: float = 0.9,
    rng: np.random.Generator | None = None,
) -> DemandModel:
    """拟合并做时段整批 cluster bootstrap。"""
    rng = rng or np.random.default_rng()
    rows = build_choice_rows(bundle)
    if not rows:
        raise ValueError("没有可用于随机轮换比较的旅程")
    params, converged, message = _fit_once(rows)
    periods = sorted({r.period for r in rows})
    segments = tuple(sorted({r.segment for r in rows}))
    model = DemandModel(
        intercepts={p: float(params[i]) for i, p in enumerate(PLANS)},
        price_coef=float(params[-1]),
        segments=segments,
        periods=tuple(periods),
        rows=rows,
        converged=converged,
        message=message,
        cl_level=cl_level,
    )
    by_period: dict[str, list[ChoiceRow]] = {}
    for row in rows:
        by_period.setdefault(row.period, []).append(row)
    boot_params = np.zeros((n_bootstrap, len(PLANS) + 1))
    period_list = list(by_period)
    for b in range(n_bootstrap):
        picked = rng.choice(
            len(period_list), size=len(period_list), replace=True
        )
        resampled = tuple(
            row for idx in picked for row in by_period[period_list[idx]]
        )
        # 热启动：重抽样数据与原样本同源，从点估计参数起步
        boot_params[b], _, _ = _fit_once(resampled, x0=params)
    model.bootstrap_params = boot_params
    return model


def evaluate_holdout(
    bundle: DataBundle, n_folds: int = 5, rng: np.random.Generator | None = None
) -> "HoldoutReport":
    """按时段整批划分折外集，评估留出期表现。"""
    rng = rng or np.random.default_rng()
    rows = build_choice_rows(bundle)
    periods = np.array(sorted({r.period for r in rows}))
    rng.shuffle(periods)
    folds = np.array_split(periods, min(n_folds, len(periods)))
    fold_reports = []
    for fold in folds:
        held = set(fold)
        train = tuple(r for r in rows if r.period not in held)
        test = tuple(r for r in rows if r.period in held)
        params, _, _ = _fit_once(train)
        base = DemandModel(
            intercepts={p: float(params[i]) for i, p in enumerate(PLANS)},
            price_coef=float(params[-1]),
            segments=(), periods=(), rows=train, converged=True, message="holdout",
        )
        log_losses, abs_errors, correct = [], [], 0
        for row in test:
            probs_map = base.choice_probabilities(
                {PLANS[i]: row.available[i] for i in range(len(PLANS))},
                {PLANS[i]: row.prices[i] for i in range(len(PLANS))},
            )
            keys = [PLANS[i] for i in range(len(PLANS)) if row.available[i]] + ["no_buy"]
            probs = np.array([probs_map[k] for k in keys])
            choice_key = (
                PLANS[row.choice] if row.choice < len(PLANS) else "no_buy"
            )
            p_actual = probs_map[choice_key]
            log_losses.append(-math.log(max(p_actual, 1e-12)))
            if keys[int(np.argmax(probs))] == choice_key:
                correct += 1
            target = np.zeros(len(keys))
            target[keys.index(choice_key)] = 1.0
            abs_errors.extend(np.abs(probs - target))
        fold_reports.append(
            FoldReport(
                periods=tuple(sorted(held)),
                n_rows=len(test),
                log_loss=float(np.mean(log_losses)),
                mean_abs_error=float(np.mean(abs_errors)),
                choice_accuracy=correct / len(test),
            )
        )
    return HoldoutReport(folds=tuple(fold_reports))


@dataclass(frozen=True)
class FoldReport:
    periods: tuple[str, ...]
    n_rows: int
    log_loss: float
    mean_abs_error: float
    choice_accuracy: float


@dataclass(frozen=True)
class HoldoutReport:
    folds: tuple[FoldReport, ...]

    @property
    def mean_log_loss(self) -> float:
        return float(np.mean([f.log_loss for f in self.folds]))

    @property
    def mean_abs_error(self) -> float:
        return float(np.mean([f.mean_abs_error for f in self.folds]))

    @property
    def mean_choice_accuracy(self) -> float:
        return float(np.mean([f.choice_accuracy for f in self.folds]))
