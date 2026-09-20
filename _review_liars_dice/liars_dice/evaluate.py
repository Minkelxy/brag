"""
自博弈评估：测试 AI 强度。
对比对象：
1. 随机玩家（Random）
2. 朴素概率玩家（Naive：叫价选最可能成立的，开骰当成立概率<0.5）
3. AI 自身（Nash，应接近 50%）
"""
from __future__ import annotations
import random
import sys
from typing import List, Tuple

sys.path.insert(0, "/workspace/liars_dice")

import numpy as np
from cfr import CFRSolver
from ai import LiarsDiceAI, OpponentModel
from dice_utils import DiceHand, effective_count, all_hands
from game import GameNode, rank_to_bid, bid_rank


def roll(n: int) -> DiceHand:
    return tuple(sorted(random.randint(1, 6) for _ in range(n)))


class RandomPlayer:
    def decide(self, node: GameNode, hand: DiceHand, player: int, max_qty: int) -> int:
        from game import raise_actions
        if node.is_root:
            acts = raise_actions(node, max_qty)
            return random.choice(acts)
        # 50% 开骰，50% 加注
        if random.random() < 0.5:
            return -1
        acts = raise_actions(node, max_qty)
        if not acts:
            return -1
        return random.choice(acts)


class NaivePlayer:
    """朴素玩家：开骰当且仅当叫价成立概率<0.5；加注选成立概率最高且>当前叫价的。"""
    def __init__(self, n_dice: int):
        self.n_dice = n_dice
        self.max_qty = n_dice * 2

    def _true_prob(self, qty, face, one_wild, hand):
        from math import comb
        my = effective_count(hand, face, one_wild and face != 1)
        need = qty - my
        if need <= 0:
            return 1.0
        p = 2.0/6 if (one_wild and face != 1) else 1.0/6
        return sum(comb(self.n_dice, k)*p**k*(1-p)**(self.n_dice-k) for k in range(need, self.n_dice+1))

    def decide(self, node: GameNode, hand: DiceHand, player: int, max_qty: int) -> int:
        from game import raise_actions
        if not node.is_root:
            qty, face = rank_to_bid(node.bid_rank)
            p_true = self._true_prob(qty, face, node.one_wild, hand)
            if p_true < 0.5:
                return -1
        # 加注：找成立概率最高的更高叫价
        best_a, best_p = None, -1.0
        for r in raise_actions(node, max_qty):
            q, f = rank_to_bid(r)
            new_wild = node.one_wild and (f != 1)
            p = self._true_prob(q, f, new_wild, hand)
            if p > best_p:
                best_p, best_a = p, r
        if best_a is None:
            return -1
        return best_a


def play_game(p0, p1, n_dice: int) -> int:
    """玩一局，返回 1 表示 p0 赢，-1 表示 p1 赢。"""
    max_qty = n_dice * 2
    h0, h1 = roll(n_dice), roll(n_dice)
    node = GameNode(-1, True, 0)
    while True:
        p = node.player
        hand = h0 if p == 0 else h1
        player = p0 if p == 0 else p1
        if isinstance(player, LiarsDiceAI):
            aid = player.decide(node, hand, p)
        else:
            aid = player.decide(node, hand, p, max_qty)
        if aid == -1:
            qty, face = rank_to_bid(node.bid_rank)
            ew = node.one_wild if face != 1 else False
            total = effective_count(h0, face, ew) + effective_count(h1, face, ew)
            bidder = 1 - p
            if total >= qty:
                return 1 if bidder == 0 else -1
            else:
                return 1 if p == 0 else -1
        else:
            q, f = rank_to_bid(aid)
            nw = node.one_wild and (f != 1)
            node = GameNode(aid, nw, 1 - p)


def evaluate(ai: LiarsDiceAI, opponent, n_dice: int, n_games: int = 1000) -> float:
    """AI 作为玩家0 vs opponent 作为玩家1。返回 AI 胜率。"""
    wins = 0
    for i in range(n_games):
        # 轮流先手
        if i % 2 == 0:
            r = play_game(ai, opponent, n_dice)
        else:
            r = play_game(opponent, ai, n_dice)
            r = -r
        if r == 1:
            wins += 1
    return wins / n_games


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/liars_dice/cfr_model.pkl")
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()

    print("加载模型...")
    solver = CFRSolver.load(args.model)
    n_dice = solver.n_dice
    ai = LiarsDiceAI(solver, exploit=False)

    print(f"\n=== AI (Nash) 强度评估（{n_dice} 颗骰子，{args.games} 局）===")

    print("\nvs 随机玩家:")
    wr = evaluate(ai, RandomPlayer(), n_dice, args.games)
    print(f"  胜率: {wr:.1%}")

    print("\nvs 朴素概率玩家:")
    wr = evaluate(ai, NaivePlayer(n_dice), n_dice, args.games)
    print(f"  胜率: {wr:.1%}")

    print("\nvs 自身 (Nash vs Nash，应接近 50%):")
    wr = evaluate(ai, LiarsDiceAI(solver, exploit=False), n_dice, args.games)
    print(f"  胜率: {wr:.1%}")


if __name__ == "__main__":
    main()
