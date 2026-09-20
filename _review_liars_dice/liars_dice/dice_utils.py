"""
骰子工具模块：枚举骰子手牌（多重集）、概率、有效点数计算。
吹牛规则：1 点为万能骰（可当作任意点数），但若有人叫过 1，则 1 不再万能。
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations_with_replacement
import math
from typing import List, Tuple

import numpy as np

Face = int  # 1..6
DiceHand = Tuple[int, ...]  # 已排序的骰子元组，如 (1,1,2,3,6)


def _multiset_probability(hand: DiceHand) -> float:
    """计算某个骰子多重集手牌的概率（5 颗骰子）。"""
    n = len(hand)
    counts = {}
    for d in hand:
        counts[d] = counts.get(d, 0) + 1
    denom = 1
    for c in counts.values():
        denom *= math.factorial(c)
    return math.factorial(n) / denom * (1.0 / 6.0) ** n


@lru_cache(maxsize=None)
def all_hands(n_dice: int) -> Tuple[DiceHand, ...]:
    """枚举 n 颗骰子的所有多重集手牌，按字典序排序。"""
    return tuple(combinations_with_replacement(range(1, 7), n_dice))


@lru_cache(maxsize=None)
def hand_probabilities(n_dice: int) -> np.ndarray:
    """返回所有手牌的概率向量，shape=(num_hands,)。"""
    hands = all_hands(n_dice)
    return np.array([_multiset_probability(h) for h in hands], dtype=np.float64)


@lru_cache(maxsize=None)
def hand_to_index(n_dice: int) -> dict:
    """手牌 -> 索引。"""
    return {h: i for i, h in enumerate(all_hands(n_dice))}


def count_face(hand: DiceHand, face: Face) -> int:
    """手牌中 face 点的数量。"""
    return hand.count(face)


def effective_count(hand: DiceHand, face: Face, one_wild: bool) -> int:
    """
    手牌中 face 的有效数量。
    one_wild=True 且 face!=1 时，1 点算作 face。
    """
    c = hand.count(face)
    if one_wild and face != 1:
        c += hand.count(1)
    return c


def effective_count_2d(hands0: List[DiceHand], hands1: List[DiceHand],
                       face: Face, one_wild: bool) -> np.ndarray:
    """
    计算两玩家所有手牌组合下，face 的有效总数矩阵。
    返回 shape=(len(hands0), len(hands1)) 的矩阵。
    """
    n0, n1 = len(hands0), len(hands1)
    # 玩家 0 的有效计数向量
    ec0 = np.array([effective_count(h, face, one_wild) for h in hands0], dtype=np.int32)
    ec1 = np.array([effective_count(h, face, one_wild) for h in hands1], dtype=np.int32)
    return ec0[:, None] + ec1[None, :]


def bid_true_matrix(hands0, hands1, qty: int, face: Face, one_wild: bool) -> np.ndarray:
    """
    返回叫价 (qty, face) 是否成立的布尔矩阵。
    成立 = 两方有效总数 >= qty。
    shape=(len(hands0), len(hands1))，dtype=bool。
    """
    total = effective_count_2d(hands0, hands1, face, one_wild)
    return total >= qty


def hand_str(hand: DiceHand) -> str:
    return "[" + " ".join(str(d) for d in hand) + "]"


if __name__ == "__main__":
    n = 5
    hands = all_hands(n)
    probs = hand_probabilities(n)
    print(f"5 颗骰子共 {len(hands)} 种手牌，概率和 = {probs.sum():.6f}")
    print("前 5 种手牌:", hands[:5])
