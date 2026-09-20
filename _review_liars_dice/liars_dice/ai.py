"""
AI 玩家：基于 CFR 纳什均衡策略 + 对手建模剥削。

策略层次：
1. 纳什均衡（CFR 平均策略）—— 保证不输，对手无法剥削。
2. 对手建模 —— 统计对手的开骰频率与叫价倾向，偏离均衡时进行剥削。
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cfr import CFRSolver
from dice_utils import DiceHand, effective_count, all_hands
from game import GameNode, bid_rank, rank_to_bid, raise_actions, num_bids


@dataclass
class OpponentModel:
    """对手行为统计，用于剥削。"""
    # 面对叫价 (qty, face) 时开骰的次数 / 总次数
    call_count: Dict[Tuple[int, int], int] = field(default_factory=dict)
    call_total: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # 对手在各 (qty, face) 上叫价的次数（粗略偏好）
    bid_count: Dict[Tuple[int, int], int] = field(default_factory=dict)
    bid_total: int = 0

    def record_call(self, bid: Tuple[int, int], called: bool) -> None:
        self.call_total[bid] = self.call_total.get(bid, 0) + 1
        if called:
            self.call_count[bid] = self.call_count.get(bid, 0) + 1

    def record_bid(self, bid: Tuple[int, int]) -> None:
        self.bid_count[bid] = self.bid_count.get(bid, 0) + 1
        self.bid_total += 1

    def call_prob(self, bid: Tuple[int, int], prior: float = 0.5) -> float:
        """带先验的开骰概率估计（Bayesian 平滑）。"""
        c = self.call_count.get(bid, 0)
        t = self.call_total.get(bid, 0)
        # 先验样本数
        prior_n = 8
        return (c + prior * prior_n) / (t + prior_n)


class LiarsDiceAI:
    def __init__(self, solver: CFRSolver, exploit: bool = True):
        self.solver = solver
        self.n_dice = solver.n_dice
        self.max_qty = solver.max_qty
        self.avg0, self.avg1 = solver.average_strategy()
        self.exploit = exploit
        self.opp_model = OpponentModel()
        # 节点索引缓存
        self._node_index = {n: i for i, n in enumerate(solver.nodes)}
        self.hands = solver.hands0
        self._hand_index = {h: i for i, h in enumerate(self.hands)}

    def _node_idx(self, node: GameNode) -> int:
        return self._node_index[node]

    def _hand_idx(self, hand: DiceHand) -> int:
        h = tuple(sorted(hand))
        return self._hand_index[h]

    def _strategy(self, node: GameNode, hand: DiceHand, player: int) -> np.ndarray:
        """返回该信息集下各动作的概率分布。"""
        idx = self._node_idx(node)
        hi = self._hand_idx(hand)
        if player == 0:
            return self.avg0[idx][hi]
        else:
            return self.avg1[idx][hi]

    def nash_action(self, node: GameNode, hand: DiceHand, player: int) -> int:
        """按纳什均衡策略采样一个动作。返回 action_id：-1 开骰，>=0 加注秩。"""
        probs = self._strategy(node, hand, player)
        actions = [a for a, _ in self.solver.node_actions[self._node_idx(node)]]
        return int(np.random.choice(actions, p=probs))

    # ---------- 对手剥削 ----------
    def _bid_true_prob(self, node: GameNode, hand: DiceHand) -> float:
        """在自己手牌已知下，当前叫价成立的概率（对手骰子均匀分布）。"""
        if node.is_root:
            return 0.0
        qty, face = rank_to_bid(node.bid_rank)
        eval_wild = node.one_wild if face != 1 else False
        my_count = effective_count(hand, face, eval_wild)
        needed = qty - my_count
        if needed <= 0:
            return 1.0
        # 对手 n_dice 颗骰子，每颗有效贡献概率 p
        if eval_wild and face != 1:
            p = 2.0 / 6.0  # 1 或 face
        else:
            p = 1.0 / 6.0
        # P(对手有效数 >= needed)
        n = self.n_dice
        from math import comb
        prob = 0.0
        for k in range(needed, n + 1):
            prob += comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        return float(prob)

    def exploit_action(self, node: GameNode, hand: DiceHand, player: int) -> int:
        """
        结合对手模型的剥削决策。
        - 开骰：当叫价成立概率 < 阈值（阈值受对手开骰倾向影响）。
        - 加注：选择期望收益最高的叫价，考虑对手开骰概率。
        """
        idx = self._node_idx(node)
        actions = self.solver.node_actions[idx]
        nash_probs = self._strategy(node, hand, player)

        # ---- 评估每个动作的期望收益 ----
        evs = []
        for a_idx, (aid, child) in enumerate(actions):
            if aid == -1:
                # 开骰：赢 iff 叫价不成立
                p_true = self._bid_true_prob(node, hand)
                ev = 1.0 - 2.0 * p_true
                evs.append(ev)
            else:
                # 加注到 bid b
                qty, face = rank_to_bid(aid)
                new_wild = node.one_wild and (face != 1)
                my_count = effective_count(hand, face, new_wild)
                needed = qty - my_count
                if needed <= 0:
                    p_true_new = 1.0
                else:
                    from math import comb
                    n = self.n_dice
                    p = 2.0 / 6.0 if (new_wild and face != 1) else 1.0 / 6.0
                    p_true_new = sum(comb(n, k) * p ** k * (1 - p) ** (n - k)
                                     for k in range(needed, n + 1))
                # 对手面对该叫价的开骰概率
                opp_call = self.opp_model.call_prob((qty, face))
                # 若对手开骰：我赢 iff 叫价成立
                ev_if_call = 2.0 * p_true_new - 1.0
                # 若对手加注：真实叫价仍占优（后续可开骰反制），虚假叫价处于劣势
                # 用 0.3 系数近似延续价值，真实叫价优于虚假叫价
                ev_if_raise = 0.3 * (2.0 * p_true_new - 1.0)
                ev = opp_call * ev_if_call + (1.0 - opp_call) * ev_if_raise
                evs.append(ev)

        evs = np.array(evs)
        # 纳什策略作为先验，向高 EV 动作倾斜
        # 温度参数：对手模型数据越多越激进
        confidence = min(1.0, self.opp_model.bid_total / 30.0)
        if confidence < 0.1:
            # 数据不足，用纳什
            return self.nash_action(node, hand, player)

        # softmax 结合 EV 与纳什先验
        logits = evs * (8.0 * confidence) + np.log(nash_probs + 1e-12)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        action_ids = [a for a, _ in actions]
        return int(np.random.choice(action_ids, p=weights))

    def decide(self, node: GameNode, hand: DiceHand, player: int) -> int:
        if self.exploit and self.opp_model.bid_total >= 5:
            return self.exploit_action(node, hand, player)
        return self.nash_action(node, hand, player)
