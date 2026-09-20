# 吹牛骰子 / Brag

吹牛骰子（Liar's Dice）的 CFR 求解与对弈实现，包含一个纯静态网页版和一套 Python 参考实现。

网页版用 CFR+ 训练出的纳什均衡策略作为对手，模型以 JavaScript 数据文件形式内嵌，无需后端。

## 在线体验

<http://117.72.184.12/damai/>

## 项目结构

```
.
├── index.html          页面入口
├── app.js              交互与对局流程
├── logic.js            游戏规则 + 策略查询
├── style.css           样式
├── cfr-model.js        CFR 策略数据（浏览器可直接加载，约 788KB）
├── train-cfr.js        Node 脚本：训练并导出 cfr-model.js
└── _review_liars_dice/
    └── liars_dice/     Python 参考实现
        ├── cfr.py          CFR 求解器（纳什均衡）
        ├── cfr_numba.py    Numba 加速的迭代内核（并行）
        ├── dice_utils.py   骰子手牌枚举、概率、有效点数
        ├── game.py         叫价空间、节点状态、叫价合法性
        ├── ai.py           AI 玩家：CFR 策略 + 对手建模剥削
        ├── play.py         命令行对弈（人类 vs AI）
        ├── evaluate.py     自博弈强度评估
        ├── train.py        训练并保存 cfr_model.pkl
        └── cfr_model.pkl   预训练模型（约 80MB）
```

## 网页版

纯静态，任意 HTTP 服务即可，无需构建：

```bash
python3 -m http.server 8000
# 打开 http://localhost:8000
```

规则为每人固定 3 颗骰子、1 点为万能点。页面支持单局 / 三局两胜 / 五局三胜 / 七局四胜。

重新训练模型（需要 Node.js）：

```bash
node train-cfr.js        # 重新生成 cfr-model.js
```

## Python 版

```bash
cd _review_liars_dice/liars_dice
pip install numpy numba

python play.py           # 命令行对弈
python train.py          # 重新训练（默认 5 颗骰子 5000 次迭代）
python evaluate.py       # 自博弈评估 AI 强度
```

仓库内已附带 `cfr_model.pkl`，`play.py` 可直接使用，无需先训练。

## 算法说明

CFR（Counterfactual Regret Minimization）通过反复自博弈迭代，最小化各信息集上的反事实遗憾，
从而逼近不完全信息博弈的纳什均衡。本项目在三人以内的叫价空间上求解，并使用 CFR+ 变体加速收敛。
Python 版的迭代内核用 Numba 并行化，网页版的策略表则离线训练后导出为静态数据。

## License

未指定。
