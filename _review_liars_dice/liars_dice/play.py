"""
交互式吹牛游戏：人类 vs AI。
"""
from __future__ import annotations
import random
import sys

sys.path.insert(0, "/workspace/liars_dice")

from cfr import CFRSolver
from ai import LiarsDiceAI
from dice_utils import DiceHand, effective_count, hand_str
from game import GameNode, bid_rank, rank_to_bid, raise_actions, num_bids


def roll_dice(n: int) -> DiceHand:
    return tuple(sorted(random.randint(1, 6) for _ in range(n)))


def print_hand(hand: DiceHand, label: str = "你的骰子") -> None:
    print(f"  {label}: {hand_str(hand)}")


def parse_bid(s: str, max_qty: int) -> tuple:
    """解析 '3个4' 或 '3 4' 格式。"""
    s = s.strip().replace("个", " ").replace(" ", " ")
    parts = s.split()
    if len(parts) != 2:
        return None
    try:
        qty = int(parts[0])
        face = int(parts[1])
    except ValueError:
        return None
    if not (1 <= face <= 6) or not (1 <= qty <= max_qty):
        return None
    return (qty, face)


def play_round(ai: LiarsDiceAI, human_first: bool, n_dice: int) -> int:
    """
    玩一轮。返回 1 表示人类赢，-1 表示 AI 赢。
    """
    max_qty = n_dice * 2
    human_hand = roll_dice(n_dice)
    ai_hand = roll_dice(n_dice)

    # 玩家编号：人类=0, AI=1（如果人类先）；否则反过来
    if human_first:
        human_player, ai_player = 0, 1
    else:
        human_player, ai_player = 1, 0

    node = GameNode(bid_rank=-1, one_wild=True, player=0)
    history = []  # 叫价历史

    print("\n" + "=" * 50)
    print(f"新一局开始！你{'先叫' if human_first else '后叫'}")
    print_hand(human_hand)

    while True:
        if node.player == human_player:
            # 人类行动
            current_bid = None if node.is_root else rank_to_bid(node.bid_rank)
            if node.is_root:
                print("\n轮到你叫价。")
            else:
                q, f = rank_to_bid(node.bid_rank)
                wild_note = "（1已叫死，不再万能）" if not node.one_wild else ""
                print(f"\n当前叫价: {q}个{f} {wild_note}")
                print("轮到你。输入 '开' 开骰，或输入新叫价如 '3个4'。")
            inp = input("> ").strip()
            if inp in ("开", "kai", "call", "d"):
                if node.is_root:
                    print("还没叫价，不能开！")
                    continue
                # 记录人类开骰行为
                ai.opp_model.record_call(current_bid, True)
                # 人类开骰
                qty, face = rank_to_bid(node.bid_rank)
                eval_wild = node.one_wild if face != 1 else False
                total = effective_count(human_hand, face, eval_wild) + effective_count(ai_hand, face, eval_wild)
                print(f"开骰！实际 {face} 点有效总数: {total}，叫价 {qty}个{face}")
                print(f"AI 的骰子: {hand_str(ai_hand)}")
                bidder = 1 - node.player
                if total >= qty:
                    print(f"叫价成立！叫价者({'AI' if bidder == ai_player else '你'})赢。")
                    winner = bidder
                else:
                    print(f"叫价不成立！开骰者({'你' if node.player == human_player else 'AI'})赢。")
                    winner = node.player
                return 1 if winner == human_player else -1
            else:
                bid = parse_bid(inp, max_qty)
                if bid is None:
                    print("无效输入，请输入如 '3个4'。")
                    continue
                qty, face = bid
                new_rank = bid_rank(qty, face, max_qty)
                if not node.is_root and new_rank <= node.bid_rank:
                    print(f"叫价必须大于当前叫价 {rank_to_bid(node.bid_rank)}。")
                    continue
                # 记录人类行为：面对叫价选择了加注（没开），以及新叫价
                if current_bid is not None:
                    ai.opp_model.record_call(current_bid, False)
                ai.opp_model.record_bid((qty, face))
                history.append(bid)
                # 更新节点
                new_wild = node.one_wild and (face != 1)
                node = GameNode(bid_rank=new_rank, one_wild=new_wild, player=ai_player)
                print(f"你叫了 {qty}个{face}")
        else:
            # AI 行动
            ai_hand_sorted = tuple(sorted(ai_hand))
            aid = ai.decide(node, ai_hand_sorted, ai_player)
            if aid == -1:
                # AI 开骰
                qty, face = rank_to_bid(node.bid_rank)
                eval_wild = node.one_wild if face != 1 else False
                total = effective_count(human_hand, face, eval_wild) + effective_count(ai_hand, face, eval_wild)
                print(f"\nAI 开骰！实际 {face} 点有效总数: {total}，叫价 {qty}个{face}")
                print(f"AI 的骰子: {hand_str(ai_hand)}")
                bidder = 1 - node.player
                if total >= qty:
                    print(f"叫价成立！叫价者({'AI' if bidder == ai_player else '你'})赢。")
                    winner = bidder
                else:
                    print(f"叫价不成立！开骰者({'你' if node.player == human_player else 'AI'})赢。")
                    winner = node.player
                return 1 if winner == human_player else -1
            else:
                qty, face = rank_to_bid(aid)
                history.append((qty, face))
                new_wild = node.one_wild and (face != 1)
                node = GameNode(bid_rank=aid, one_wild=new_wild, player=human_player)
                print(f"\nAI 叫了 {qty}个{face}")
                if face == 1 and node.one_wild is False:
                    pass  # 1 叫死了


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/liars_dice/cfr_model.pkl",
                        help="CFR 模型路径")
    parser.add_argument("--dice", type=int, default=5)
    parser.add_argument("--no-exploit", action="store_true", help="禁用对手剥削（纯纳什）")
    args = parser.parse_args()

    print("正在加载 CFR 模型...")
    try:
        solver = CFRSolver.load(args.model)
        print(f"已加载模型（{solver.n_dice} 颗骰子）。")
    except FileNotFoundError:
        print(f"未找到模型 {args.model}，请先运行 train.py 训练。")
        print("或运行: python train.py --dice 5 --iter 5000")
        sys.exit(1)

    if solver.n_dice != args.dice:
        print(f"警告：模型为 {solver.n_dice} 颗骰子，当前使用 {args.dice}。将使用模型设置。")

    ai = LiarsDiceAI(solver, exploit=not args.no_exploit)
    n_dice = solver.n_dice

    human_score = 0
    ai_score = 0
    round_num = 0
    human_first = True

    print("\n" + "=" * 50)
    print("  吹牛骰子游戏 · 人类 vs AI（纳什均衡 + 剥削）")
    print("  规则：1点万能，叫过1后1不再万能")
    print("  输入：'3个4' 叫价，'开' 开骰")
    print("=" * 50)

    while True:
        round_num += 1
        print(f"\n--- 第 {round_num} 局 (你 {human_score} : {ai_score} AI) ---")
        result = play_round(ai, human_first, n_dice)
        if result == 1:
            human_score += 1
            print("👉 你赢了这局！")
        else:
            ai_score += 1
            print("👉 AI 赢了这局。")

        human_first = not human_first
        # 记录对手模型（人类行为）
        # 简单记录：上一局的人类叫价和开骰行为已在 decide 时通过 opp_model 记录
        # 这里可以额外统计

        again = input("\n再来一局？(y/n) ").strip().lower()
        if again not in ("y", "yes", "是", ""):
            break

    print(f"\n最终比分：你 {human_score} : {ai_score} AI")
    print("再见！")


if __name__ == "__main__":
    main()
