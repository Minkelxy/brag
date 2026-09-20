"""
训练脚本：运行 CFR 并保存纳什均衡策略。
用法: python train.py --dice 5 --iter 5000 --out cfr_model.pkl
"""
from __future__ import annotations
import argparse
import time

from cfr import CFRSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dice", type=int, default=5, help="每人骰子数")
    parser.add_argument("--iter", type=int, default=5000, help="CFR 迭代次数")
    parser.add_argument("--out", default="/workspace/liars_dice/cfr_model.pkl")
    args = parser.parse_args()

    print(f"初始化 CFR 求解器（{args.dice} 颗骰子）...")
    t0 = time.time()
    solver = CFRSolver(n_dice=args.dice)
    print(f"  节点数: {solver.n_nodes}")
    print(f"  手牌数: {solver.n_hands}")
    print(f"  初始化耗时: {time.time()-t0:.1f}s")

    print(f"\n开始训练 {args.iter} 次迭代...")
    t0 = time.time()
    solver.train(iterations=args.iter, verbose=True)
    elapsed = time.time() - t0
    print(f"\n训练完成！耗时 {elapsed:.1f}s ({elapsed/args.iter*1000:.1f}ms/iter)")

    solver.save(args.out)
    print("模型已保存。")


if __name__ == "__main__":
    main()
