"""
游戏规则模块：叫价空间、节点状态、叫价合法性。

叫价 (qty, face)：qty 为猜测的颗数，face 为点数(1..6)。
叫价全序：(q1,f1) < (q2,f2) 当且仅当 q1<q2 或 (q1==q2 and f1<f2)。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from dice_utils import DiceHand, effective_count_2d, all_hands

Bid = Tuple[int, int]  # (qty, face)


def bid_rank(qty: int, face: int, max_qty: int) -> int:
    """将叫价映射为全序秩 0..(max_qty*6-1)。"""
    return (qty - 1) * 6 + (face - 1)


def rank_to_bid(rank: int) -> Bid:
    qty = rank // 6 + 1
    face = rank % 6 + 1
    return (qty, face)


def num_bids(max_qty: int) -> int:
    return max_qty * 6


def all_valid_bids(max_qty: int) -> List[Bid]:
    return [(q, f) for q in range(1, max_qty + 1) for f in range(1, 7)]


@dataclass(frozen=True)
class GameNode:
    """
    博弈树节点（仅与叫价结构有关，与骰子无关）。
    - bid_rank: 当前叫价的秩；-1 表示还没叫价（根节点）。
    - one_wild: 当前 1 是否仍为万能（叫过 1 后变为 False）。
    - player: 轮到谁行动（0 或 1）。
    """
    bid_rank: int
    one_wild: bool
    player: int

    @property
    def is_root(self) -> bool:
        return self.bid_rank == -1

    def bid(self, max_qty: int) -> Optional[Bid]:
        if self.is_root:
            return None
        return rank_to_bid(self.bid_rank)


def raise_actions(node: GameNode, max_qty: int) -> List[int]:
    """从当前节点出发，所有合法的加注叫价秩列表（严格大于当前叫价）。"""
    start = node.bid_rank + 1
    return list(range(start, num_bids(max_qty)))


def next_node(node: GameNode, raise_rank: int, max_qty: int) -> GameNode:
    """加注到 raise_rank 后的新节点。"""
    qty, face = rank_to_bid(raise_rank)
    new_wild = node.one_wild and (face != 1)
    return GameNode(bid_rank=raise_rank, one_wild=new_wild, player=1 - node.player)


def call_payoff_matrix(node: GameNode, hands0: List[DiceHand],
                       hands1: List[DiceHand], max_qty: int) -> np.ndarray:
    """
    当 node.player 选择"开"时，玩家 0 的收益矩阵。
    收益：+1 玩家 0 赢，-1 玩家 0 输。
    规则：若叫价成立则叫价者赢，否则开骰者赢。
    """
    assert not node.is_root, "根节点不能开骰"
    qty, face = rank_to_bid(node.bid_rank)
    # 评估该叫价时使用的 wild 状态：叫价之前的 wild
    # 若 face != 1：叫价前 wild = 当前 wild
    # 若 face == 1：wild 不影响（1 只算 1）
    eval_wild = node.one_wild if face != 1 else False
    bid_true = effective_count_2d(hands0, hands1, face, eval_wild) >= qty
    # 叫价者 = 1 - node.player；开骰者 = node.player
    # bidder wins iff bid_true
    # player0 wins iff (bidder==0 and bid_true) or (bidder==1 and not bid_true)
    bidder = 1 - node.player
    player0_wins = (bid_true & (bidder == 0)) | (~bid_true & (bidder == 1))
    payoff = np.where(player0_wins, 1.0, -1.0)
    return payoff


def build_node_order(max_qty: int) -> List[GameNode]:
    """
    以拓扑序（从根到叶）列出所有叫价节点。
    反向遍历用于价值回溯。
    """
    nodes = [GameNode(-1, True, 0)]  # 根
    # BFS 扩展
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        idx += 1
        for r in raise_actions(node, max_qty):
            child = next_node(node, r, max_qty)
            if child not in nodes:
                nodes.append(child)
    return nodes


def build_children_map(nodes: List[GameNode], max_qty: int) -> dict:
    """node -> {action_rank: child_node}。action_rank=-1 表示开骰，>=0 表示加注秩。"""
    nidx = {n: i for i, n in enumerate(nodes)}
    children = {}
    for i, node in enumerate(nodes):
        kids = {}
        if not node.is_root:
            kids[-1] = None  # 开骰 -> 终止
        for r in raise_actions(node, max_qty):
            child = next_node(node, r, max_qty)
            kids[r] = nidx[child]
        children[i] = kids
    return children


if __name__ == "__main__":
    max_qty = 10
    nodes = build_node_order(max_qty)
    print(f"总节点数: {len(nodes)}")
    print(f"叫价数: {num_bids(max_qty)}")
