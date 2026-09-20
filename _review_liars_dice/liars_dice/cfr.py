"""
CFR（反事实遗憾最小化）求解器，用于计算吹牛博弈的纳什均衡策略。

核心思想：
- 两人零和不完全信息博弈，CFR 收敛到纳什均衡。
- 博弈树（叫价结构）与骰子无关，仅终止收益依赖骰子。
- 用 numpy 向量化：价值 V[i][d0,d1]、到达概率 reach0/reach1[i][d0,d1]。
- 采用 Regret Matching+ (RM+) 加速收敛。
"""
from __future__ import annotations
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from dice_utils import all_hands, hand_probabilities, DiceHand
from game import (
    GameNode, build_node_order, build_children_map,
    call_payoff_matrix, rank_to_bid, num_bids, raise_actions,
)


class CFRSolver:
    def __init__(self, n_dice: int = 5):
        self.n_dice = n_dice
        self.max_qty = n_dice * 2
        self.hands0: List[DiceHand] = list(all_hands(n_dice))
        self.hands1: List[DiceHand] = list(all_hands(n_dice))
        self.n_hands = len(self.hands0)
        self.P0 = hand_probabilities(n_dice)  # (n_hands,)
        self.P1 = hand_probabilities(n_dice)

        self.nodes: List[GameNode] = build_node_order(self.max_qty)
        self.n_nodes = len(self.nodes)
        self.children: Dict[int, Dict[int, Optional[int]]] = build_children_map(self.nodes, self.max_qty)
        # node_idx -> list of (action_id, child_idx or None for call)
        self.node_actions: List[List[Tuple[int, Optional[int]]]] = []
        for i in range(self.n_nodes):
            acts = []
            if not self.nodes[i].is_root:
                acts.append((-1, None))  # 开骰
            for r in raise_actions(self.nodes[i], self.max_qty):
                acts.append((r, self.children[i][r]))
            self.node_actions.append(acts)

        # 策略数据：每个节点一个 (n_hands, n_actions) 数组
        self.sigma0: List[np.ndarray] = []
        self.sigma1: List[np.ndarray] = []
        self.regret0: List[np.ndarray] = []
        self.regret1: List[np.ndarray] = []
        self.strat_sum0: List[np.ndarray] = []
        self.strat_sum1: List[np.ndarray] = []
        self.reach_sum0: List[np.ndarray] = []  # (n_hands,) per node, for player 0 info sets
        self.reach_sum1: List[np.ndarray] = []

        for i in range(self.n_nodes):
            na = len(self.node_actions[i])
            self.sigma0.append(np.full((self.n_hands, na), 1.0 / na))
            self.sigma1.append(np.full((self.n_hands, na), 1.0 / na))
            self.regret0.append(np.zeros((self.n_hands, na)))
            self.regret1.append(np.zeros((self.n_hands, na)))
            self.strat_sum0.append(np.zeros((self.n_hands, na)))
            self.strat_sum1.append(np.zeros((self.n_hands, na)))
            self.reach_sum0.append(np.zeros(self.n_hands))
            self.reach_sum1.append(np.zeros(self.n_hands))

        # 预计算每个节点开骰的收益矩阵（玩家 0 视角）
        self.call_payoff: List[Optional[np.ndarray]] = []
        for i in range(self.n_nodes):
            if self.nodes[i].is_root:
                self.call_payoff.append(None)
            else:
                self.call_payoff.append(
                    call_payoff_matrix(self.nodes[i], self.hands0, self.hands1, self.max_qty)
                )

        # 预计算每个动作的子节点索引（-1 表示开骰/终止）和是否开骰
        self._child_idx: List[np.ndarray] = []
        self._is_call: List[np.ndarray] = []
        self._raise_pairs: List[List[Tuple[int, int]]] = []  # (a_idx, child_idx) 仅加注动作
        self._raise_aidx: List[np.ndarray] = []  # 加注动作的 a_idx 数组
        self._raise_cidx: List[np.ndarray] = []  # 加注动作的 child_idx 数组
        for i in range(self.n_nodes):
            acts = self.node_actions[i]
            cidx = np.array([c if c is not None else -1 for _, c in acts], dtype=np.int64)
            iscall = np.array([aid == -1 for aid, _ in acts], dtype=bool)
            self._child_idx.append(cidx)
            self._is_call.append(iscall)
            rp = [(a, c) for a, (aid, c) in enumerate(acts) if aid != -1]
            self._raise_pairs.append(rp)
            if rp:
                self._raise_aidx.append(np.array([a for a, _ in rp], dtype=np.int64))
                self._raise_cidx.append(np.array([c for _, c in rp], dtype=np.int64))
            else:
                self._raise_aidx.append(np.array([], dtype=np.int64))
                self._raise_cidx.append(np.array([], dtype=np.int64))

    # ---------- 策略更新（Regret Matching+） ----------
    def _update_strategy(self, regret: np.ndarray, sigma: np.ndarray) -> None:
        """原地根据累计遗憾更新策略（RM+）。"""
        pos = np.maximum(regret, 0.0)
        total = pos.sum(axis=1, keepdims=True)
        # 全为 0 时均匀随机
        uniform = np.full_like(sigma, 1.0 / sigma.shape[1])
        sigma[:] = np.where(total > 0, pos / np.maximum(total, 1e-30), uniform)

    def train(self, iterations: int, verbose: bool = True) -> None:
        """运行 CFR 训练。优先使用 numba 加速。"""
        try:
            self.train_numba(iterations, verbose)
        except Exception as e:
            if verbose:
                print(f"[CFR] numba 不可用或出错，回退到 numpy 版本: {e}")
            self.train_numpy(iterations, verbose)

    def _build_padded(self):
        """构建 numba 用的定长填充数组。"""
        n = self.n_hands
        n_nodes = self.n_nodes
        max_A = max(len(acts) for acts in self.node_actions)

        action_child = np.full((n_nodes, max_A), -1, dtype=np.int64)
        action_is_call = np.zeros((n_nodes, max_A), dtype=bool)
        action_valid = np.zeros((n_nodes, max_A), dtype=bool)
        node_player = np.array([nd.player for nd in self.nodes], dtype=np.int64)

        sigma0 = np.zeros((n_nodes, n, max_A))
        sigma1 = np.zeros((n_nodes, n, max_A))
        regret0 = np.zeros((n_nodes, n, max_A))
        regret1 = np.zeros((n_nodes, n, max_A))
        strat_sum0 = np.zeros((n_nodes, n, max_A))
        strat_sum1 = np.zeros((n_nodes, n, max_A))
        reach_sum0 = np.zeros((n_nodes, n))
        reach_sum1 = np.zeros((n_nodes, n))
        call_payoff = np.zeros((n_nodes, n, n))

        for i in range(n_nodes):
            acts = self.node_actions[i]
            for a_idx, (aid, child) in enumerate(acts):
                action_valid[i, a_idx] = True
                if aid == -1:
                    action_is_call[i, a_idx] = True
                    call_payoff[i] = self.call_payoff[i]
                else:
                    action_child[i, a_idx] = child
            na = len(acts)
            sigma0[i, :, :na] = self.sigma0[i]
            sigma1[i, :, :na] = self.sigma1[i]

        V = np.zeros((n_nodes, n, n))
        reach0 = np.zeros((n_nodes, n, n))
        reach1 = np.zeros((n_nodes, n, n))
        return (V, reach0, reach1, sigma0, sigma1, regret0, regret1,
                strat_sum0, strat_sum1, reach_sum0, reach_sum1, call_payoff,
                action_child, action_is_call, action_valid, node_player, max_A)

    def train_numba(self, iterations: int, verbose: bool = True) -> None:
        """使用 numba JIT 加速的 CFR 训练。"""
        from cfr_numba import cfr_iteration, update_strategies

        (V, reach0, reach1, sigma0, sigma1, regret0, regret1,
         strat_sum0, strat_sum1, reach_sum0, reach_sum1, call_payoff,
         action_child, action_is_call, action_valid, node_player, max_A) = self._build_padded()

        n = self.n_hands
        n_nodes = self.n_nodes
        P0 = self.P0
        P1 = self.P1

        t0 = time.time()
        for it in range(1, iterations + 1):
            cfr_iteration(
                V, reach0, reach1, sigma0, sigma1, regret0, regret1,
                strat_sum0, strat_sum1, reach_sum0, reach_sum1, call_payoff,
                action_child, action_is_call, action_valid, node_player,
                P0, P1, n_nodes, n, max_A,
            )
            update_strategies(
                regret0, regret1, sigma0, sigma1, action_valid,
                n_nodes, n, max_A,
            )
            if verbose and (it % max(1, iterations // 20) == 0 or it == 1):
                elapsed = time.time() - t0
                root_val = float(V[0].sum())
                print(f"[CFR] iter {it}/{iterations} | {elapsed:.1f}s | root_val~{root_val:.4f}")

        # 回写到 list 结构
        self.V = [V[i] for i in range(n_nodes)]
        for i in range(n_nodes):
            na = len(self.node_actions[i])
            self.sigma0[i] = sigma0[i, :, :na].copy()
            self.sigma1[i] = sigma1[i, :, :na].copy()
            self.regret0[i] = regret0[i, :, :na].copy()
            self.regret1[i] = regret1[i, :, :na].copy()
            self.strat_sum0[i] = strat_sum0[i, :, :na].copy()
            self.strat_sum1[i] = strat_sum1[i, :, :na].copy()
            self.reach_sum0[i] = reach_sum0[i].copy()
            self.reach_sum1[i] = reach_sum1[i].copy()

    def train_numpy(self, iterations: int, verbose: bool = True) -> None:
        """numpy 版本的 CFR 训练（numba 不可用时的回退）。"""
        n = self.n_hands
        n_nodes = self.n_nodes
        P0 = self.P0
        P1 = self.P1

        # 价值矩阵 V[i] (n, n)
        self.V: List[np.ndarray] = [np.zeros((n, n)) for _ in range(n_nodes)]
        reach0 = [np.zeros((n, n)) for _ in range(n_nodes)]
        reach1 = [np.zeros((n, n)) for _ in range(n_nodes)]
        # 每个节点的 V_stack (A, n, n)，后向计算后供遗憾更新复用
        v_stacks: List[Optional[np.ndarray]] = [None] * n_nodes

        t0 = time.time()
        for it in range(1, iterations + 1):
            # ---- 前向：到达概率 ----
            for i in range(n_nodes):
                reach0[i].fill(0.0)
                reach1[i].fill(0.0)
            reach0[0][:] = 1.0
            reach1[0][:] = 1.0

            for i in range(n_nodes):
                p = self.nodes[i].player
                r0 = reach0[i]
                r1 = reach1[i]
                raidx = self._raise_aidx[i]
                rcidx = self._raise_cidx[i]
                if len(raidx) == 0:
                    continue
                if p == 0:
                    sig_r = self.sigma0[i][:, raidx]  # (n, A_raise)
                    # contrib0[a, d0, d1] = r0[d0, d1] * sig_r[d0, a]
                    contrib = r0[None, :, :] * sig_r.T[:, :, None]  # (A, n, n)
                    for k in range(len(raidx)):
                        c = rcidx[k]
                        reach0[c] += contrib[k]
                        reach1[c] += r1
                else:
                    sig_r = self.sigma1[i][:, raidx]  # (n, A_raise)
                    contrib = r1[None, :, :] * sig_r.T[:, None, :]  # (A, n, n)
                    for k in range(len(raidx)):
                        c = rcidx[k]
                        reach1[c] += contrib[k]
                        reach0[c] += r0

            # ---- 后向：价值 V ----
            for i in range(n_nodes - 1, -1, -1):
                p = self.nodes[i].player
                cidx = self._child_idx[i]
                iscall = self._is_call[i]
                A = len(cidx)
                V_stack = np.empty((A, n, n))
                if iscall.any():
                    cp = self.call_payoff[i]
                for a in range(A):
                    if iscall[a]:
                        V_stack[a] = cp
                    else:
                        V_stack[a] = self.V[cidx[a]]
                v_stacks[i] = V_stack
                if p == 0:
                    self.V[i] = np.einsum('da,ade->de', self.sigma0[i], V_stack)
                else:
                    self.V[i] = np.einsum('ea,ade->de', self.sigma1[i], V_stack)

            # ---- 遗憾与策略累计 ----
            for i in range(n_nodes):
                p = self.nodes[i].player
                Vi = self.V[i]
                V_stack = v_stacks[i]
                if p == 0:
                    weighted_r1 = reach1[i] * P1[None, :]  # (n, n)
                    pi_neg0 = weighted_r1.sum(axis=1)  # (n,)
                    diff = V_stack - Vi[None, :, :]  # (A, n, n)
                    regret = np.einsum('de,ade->ad', weighted_r1, diff)  # (A, n)
                    self.regret0[i] += regret.T  # (n, A)
                    self.strat_sum0[i] += pi_neg0[:, None] * self.sigma0[i]
                    self.reach_sum0[i] += pi_neg0
                else:
                    weighted_r0 = reach0[i] * P0[:, None]  # (n, n)
                    pi_neg1 = weighted_r0.sum(axis=0)  # (n,)
                    diff = Vi[None, :, :] - V_stack  # (A, n, n) 玩家1遗憾 = Vi - Vc
                    regret = np.einsum('de,ade->ae', weighted_r0, diff)  # (A, n)
                    self.regret1[i] += regret.T  # (n, A)
                    self.strat_sum1[i] += pi_neg1[:, None] * self.sigma1[i]
                    self.reach_sum1[i] += pi_neg1

            # ---- 更新策略（RM+） ----
            for i in range(n_nodes):
                np.maximum(self.regret0[i], 0.0, out=self.regret0[i])
                np.maximum(self.regret1[i], 0.0, out=self.regret1[i])
                self._update_strategy(self.regret0[i], self.sigma0[i])
                self._update_strategy(self.regret1[i], self.sigma1[i])

            if verbose and (it % max(1, iterations // 20) == 0 or it == 1):
                elapsed = time.time() - t0
                root_val = float(self.V[0].sum())
                print(f"[CFR] iter {it}/{iterations} | {elapsed:.1f}s | root_val~{root_val:.4f}")

    def average_strategy(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """输出平均策略（纳什均衡近似）。"""
        avg0, avg1 = [], []
        for i in range(self.n_nodes):
            denom0 = self.reach_sum0[i][:, None]
            s0 = np.where(denom0 > 1e-30, self.strat_sum0[i] / np.maximum(denom0, 1e-30),
                          1.0 / self.sigma0[i].shape[1])
            avg0.append(s0 / s0.sum(axis=1, keepdims=True))

            denom1 = self.reach_sum1[i][:, None]
            s1 = np.where(denom1 > 1e-30, self.strat_sum1[i] / np.maximum(denom1, 1e-30),
                          1.0 / self.sigma1[i].shape[1])
            avg1.append(s1 / s1.sum(axis=1, keepdims=True))
        return avg0, avg1

    def save(self, path: str) -> None:
        data = {
            "n_dice": self.n_dice,
            "sigma0": self.sigma0,
            "sigma1": self.sigma1,
            "regret0": self.regret0,
            "regret1": self.regret1,
            "strat_sum0": self.strat_sum0,
            "strat_sum1": self.strat_sum1,
            "reach_sum0": self.reach_sum0,
            "reach_sum1": self.reach_sum1,
            "nodes": self.nodes,
            "node_actions": self.node_actions,
            "children": self.children,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[CFR] 策略已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> "CFRSolver":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.n_dice = data["n_dice"]
        obj.max_qty = obj.n_dice * 2
        obj.hands0 = list(all_hands(obj.n_dice))
        obj.hands1 = list(all_hands(obj.n_dice))
        obj.n_hands = len(obj.hands0)
        obj.P0 = hand_probabilities(obj.n_dice)
        obj.P1 = hand_probabilities(obj.n_dice)
        obj.nodes = data["nodes"]
        obj.n_nodes = len(obj.nodes)
        obj.children = data["children"]
        obj.node_actions = data["node_actions"]
        obj.sigma0 = data["sigma0"]
        obj.sigma1 = data["sigma1"]
        obj.regret0 = data["regret0"]
        obj.regret1 = data["regret1"]
        obj.strat_sum0 = data["strat_sum0"]
        obj.strat_sum1 = data["strat_sum1"]
        obj.reach_sum0 = data["reach_sum0"]
        obj.reach_sum1 = data["reach_sum1"]
        obj.call_payoff = []
        for i in range(obj.n_nodes):
            if obj.nodes[i].is_root:
                obj.call_payoff.append(None)
            else:
                obj.call_payoff.append(
                    call_payoff_matrix(obj.nodes[i], obj.hands0, obj.hands1, obj.max_qty)
                )
        return obj
