"""
Numba 加速的 CFR 迭代内核（并行版）。
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def cfr_iteration(
    V, reach0, reach1,
    sigma0, sigma1,
    regret0, regret1,
    strat_sum0, strat_sum1,
    reach_sum0, reach_sum1,
    call_payoff,
    action_child, action_is_call, action_valid,
    node_player,
    P0, P1,
    n_nodes, n, max_A,
):
    # ---------- 前向：到达概率 ----------
    for i in range(n_nodes):
        for d0 in range(n):
            for d1 in range(n):
                reach0[i, d0, d1] = 0.0
                reach1[i, d0, d1] = 0.0
    for d0 in range(n):
        for d1 in range(n):
            reach0[0, d0, d1] = 1.0
            reach1[0, d0, d1] = 1.0

    for i in range(n_nodes):
        p = node_player[i]
        if p == 0:
            for d0 in prange(n):
                for a in range(max_A):
                    if not action_valid[i, a] or action_is_call[i, a]:
                        continue
                    c = action_child[i, a]
                    s = sigma0[i, d0, a]
                    if s != 0.0:
                        for d1 in range(n):
                            reach0[c, d0, d1] += reach0[i, d0, d1] * s
                            reach1[c, d0, d1] += reach1[i, d0, d1]
        else:
            for d1 in prange(n):
                for a in range(max_A):
                    if not action_valid[i, a] or action_is_call[i, a]:
                        continue
                    c = action_child[i, a]
                    s = sigma1[i, d1, a]
                    if s != 0.0:
                        for d0 in range(n):
                            reach1[c, d0, d1] += reach1[i, d0, d1] * s
                            reach0[c, d0, d1] += reach0[i, d0, d1]

    # ---------- 后向：价值 V ----------
    for i in range(n_nodes - 1, -1, -1):
        p = node_player[i]
        if p == 0:
            for d0 in prange(n):
                for d1 in range(n):
                    val = 0.0
                    for a in range(max_A):
                        if not action_valid[i, a]:
                            continue
                        if action_is_call[i, a]:
                            vc = call_payoff[i, d0, d1]
                        else:
                            vc = V[action_child[i, a], d0, d1]
                        val += sigma0[i, d0, a] * vc
                    V[i, d0, d1] = val
        else:
            for d1 in prange(n):
                for d0 in range(n):
                    val = 0.0
                    for a in range(max_A):
                        if not action_valid[i, a]:
                            continue
                        if action_is_call[i, a]:
                            vc = call_payoff[i, d0, d1]
                        else:
                            vc = V[action_child[i, a], d0, d1]
                        val += sigma1[i, d1, a] * vc
                    V[i, d0, d1] = val

    # ---------- 遗憾与策略累计 ----------
    for i in prange(n_nodes):
        p = node_player[i]
        if p == 0:
            for d0 in range(n):
                pi = 0.0
                for d1 in range(n):
                    pi += P1[d1] * reach1[i, d0, d1]
                reach_sum0[i, d0] += pi
                for d1 in range(n):
                    w = P1[d1] * reach1[i, d0, d1]
                    Vi = V[i, d0, d1]
                    for a in range(max_A):
                        if not action_valid[i, a]:
                            continue
                        if action_is_call[i, a]:
                            vc = call_payoff[i, d0, d1]
                        else:
                            vc = V[action_child[i, a], d0, d1]
                        regret0[i, d0, a] += w * (vc - Vi)
                        strat_sum0[i, d0, a] += pi * sigma0[i, d0, a]
        else:
            for d1 in range(n):
                pi = 0.0
                for d0 in range(n):
                    pi += P0[d0] * reach0[i, d0, d1]
                reach_sum1[i, d1] += pi
                for d0 in range(n):
                    w = P0[d0] * reach0[i, d0, d1]
                    Vi = V[i, d0, d1]
                    for a in range(max_A):
                        if not action_valid[i, a]:
                            continue
                        if action_is_call[i, a]:
                            vc = call_payoff[i, d0, d1]
                        else:
                            vc = V[action_child[i, a], d0, d1]
                        regret1[i, d1, a] += w * (Vi - vc)
                        strat_sum1[i, d1, a] += pi * sigma1[i, d1, a]


@njit(cache=True, fastmath=True, parallel=True)
def update_strategies(regret0, regret1, sigma0, sigma1, action_valid, n_nodes, n, max_A):
    for i in range(n_nodes):
        cnt = 0
        for a in range(max_A):
            if action_valid[i, a]:
                cnt += 1
        inv_cnt = 1.0 / cnt if cnt > 0 else 0.0

        for d in prange(n):
            s = 0.0
            for a in range(max_A):
                r = regret0[i, d, a]
                if r < 0.0:
                    regret0[i, d, a] = 0.0
                    r = 0.0
                s += r
            if s > 0.0:
                inv_s = 1.0 / s
                for a in range(max_A):
                    sigma0[i, d, a] = regret0[i, d, a] * inv_s
            else:
                for a in range(max_A):
                    sigma0[i, d, a] = inv_cnt if action_valid[i, a] else 0.0

            s = 0.0
            for a in range(max_A):
                r = regret1[i, d, a]
                if r < 0.0:
                    regret1[i, d, a] = 0.0
                    r = 0.0
                s += r
            if s > 0.0:
                inv_s = 1.0 / s
                for a in range(max_A):
                    sigma1[i, d, a] = regret1[i, d, a] * inv_s
            else:
                for a in range(max_A):
                    sigma1[i, d, a] = inv_cnt if action_valid[i, a] else 0.0
