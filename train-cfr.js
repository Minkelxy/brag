"use strict";

// Generates a compact CFR+ policy for the finalized three-dice, always-wild rules.
// The output is a browser-safe JavaScript data file rather than a pickle model.

const DICE = 3;
const MAX_QTY = DICE * 2;
const RANK_COUNT = MAX_QTY * 6;
const HAND_COUNT = 56;
const DEFAULT_ITERATIONS = 1800;

function combinationsWithReplacement(length, minFace, hand, output) {
  if (hand.length === length) {
    output.push(hand.slice());
    return;
  }
  for (let face = minFace; face <= 6; face += 1) {
    hand.push(face);
    combinationsWithReplacement(length, face, hand, output);
    hand.pop();
  }
}

function factorial(n) {
  let value = 1;
  for (let i = 2; i <= n; i += 1) value *= i;
  return value;
}

function handProbability(hand) {
  const counts = new Map();
  hand.forEach((face) => counts.set(face, (counts.get(face) || 0) + 1));
  let denominator = 1;
  counts.forEach((count) => { denominator *= factorial(count); });
  return factorial(DICE) / denominator / Math.pow(6, DICE);
}

function bidFromRank(rank) {
  return {
    quantity: Math.floor(rank / 6) + 1,
    face: rank % 6 + 1
  };
}

function stateIndex(rank, player) {
  return rank < 0 ? 0 : 1 + rank * 2 + player;
}

function positiveRegretStrategy(regrets, sigma, actions, handOffset) {
  let total = 0;
  for (let a = 0; a < actions.length; a += 1) {
    const value = Math.max(0, regrets[handOffset + a]);
    regrets[handOffset + a] = value;
    total += value;
  }
  if (total <= 1e-15) {
    const uniform = 1 / actions.length;
    for (let a = 0; a < actions.length; a += 1) sigma[handOffset + a] = uniform;
    return;
  }
  for (let a = 0; a < actions.length; a += 1) {
    sigma[handOffset + a] = Math.max(0, regrets[handOffset + a]) / total;
  }
}

function quantizePolicy(policy) {
  return policy.map((node) => node.map((handPolicy) => {
    const quantized = handPolicy.map((probability) => Math.round(probability * 1000000));
    const total = quantized.reduce((sum, value) => sum + value, 0);
    if (total === 0) quantized[0] = 1000000;
    else if (total !== 1000000) quantized[quantized.length - 1] += 1000000 - total;
    return quantized;
  }));
}

function train(iterations) {
  const hands = [];
  combinationsWithReplacement(DICE, 1, [], hands);
  if (hands.length !== HAND_COUNT) throw new Error("Unexpected hand count");
  const handProbabilities = hands.map(handProbability);
  const handCounts = hands.map((hand) => Array.from({ length: 6 }, (_, faceIndex) => {
    const face = faceIndex + 1;
    return face === 1
      ? hand.filter((die) => die === 1).length
      : hand.filter((die) => die === face || die === 1).length;
  }));

  const actions = [{ actions: Array.from({ length: RANK_COUNT }, (_, rank) => rank), player: 0, rank: -1 }];
  for (let rank = 0; rank < RANK_COUNT; rank += 1) {
    for (let player = 0; player < 2; player += 1) {
      actions.push({ actions: [-1].concat(Array.from({ length: RANK_COUNT - rank - 1 }, (_, index) => rank + index + 1)), player, rank });
    }
  }

  const sigma0 = actions.map((node) => {
    const values = new Float64Array(HAND_COUNT * node.actions.length);
    for (let h = 0; h < HAND_COUNT; h += 1) {
      for (let a = 0; a < node.actions.length; a += 1) values[h * node.actions.length + a] = 1 / node.actions.length;
    }
    return values;
  });
  const sigma1 = actions.map((node) => sigma0[actions.indexOf(node)].slice());
  const regret0 = sigma0.map((values) => new Float64Array(values.length));
  const regret1 = sigma0.map((values) => new Float64Array(values.length));
  const strategySum0 = sigma0.map((values) => new Float64Array(values.length));
  const strategySum1 = sigma0.map((values) => new Float64Array(values.length));
  const ownReachSum0 = actions.map(() => new Float64Array(HAND_COUNT));
  const ownReachSum1 = actions.map(() => new Float64Array(HAND_COUNT));
  const reach0 = actions.map(() => new Float64Array(HAND_COUNT * HAND_COUNT));
  const reach1 = actions.map(() => new Float64Array(HAND_COUNT * HAND_COUNT));
  const values = actions.map(() => new Float64Array(HAND_COUNT * HAND_COUNT));
  const callPayoff = actions.map((node) => {
    if (node.rank < 0) return null;
    const bid = bidFromRank(node.rank);
    const payoff = new Float64Array(HAND_COUNT * HAND_COUNT);
    for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
      for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
        const total = handCounts[h0][bid.face - 1] + handCounts[h1][bid.face - 1];
        const bidTrue = total >= bid.quantity;
        const bidder = 1 - node.player;
        const player0Wins = (bidder === 0 && bidTrue) || (bidder === 1 && !bidTrue);
        payoff[h0 * HAND_COUNT + h1] = player0Wins ? 1 : -1;
      }
    }
    return payoff;
  });

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    reach0.forEach((matrix) => matrix.fill(0));
    reach1.forEach((matrix) => matrix.fill(0));
    reach0[0].fill(1);
    reach1[0].fill(1);

    for (let nodeIndex = 0; nodeIndex < actions.length; nodeIndex += 1) {
      const node = actions[nodeIndex];
      const currentReach0 = reach0[nodeIndex];
      const currentReach1 = reach1[nodeIndex];
      for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
        const rank = node.actions[actionIndex];
        if (rank < 0) continue;
        const childIndex = stateIndex(rank, 1 - node.player);
        if (node.player === 0) {
          for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
            const probability = sigma0[nodeIndex][h0 * node.actions.length + actionIndex];
            for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
              const cell = h0 * HAND_COUNT + h1;
              reach0[childIndex][cell] += currentReach0[cell] * probability;
              reach1[childIndex][cell] += currentReach1[cell];
            }
          }
        } else {
          for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
            const probability = sigma1[nodeIndex][h1 * node.actions.length + actionIndex];
            for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
              const cell = h0 * HAND_COUNT + h1;
              reach1[childIndex][cell] += currentReach1[cell] * probability;
              reach0[childIndex][cell] += currentReach0[cell];
            }
          }
        }
      }
    }

    for (let nodeIndex = actions.length - 1; nodeIndex >= 0; nodeIndex -= 1) {
      const node = actions[nodeIndex];
      const value = values[nodeIndex];
      value.fill(0);
      for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
        const rank = node.actions[actionIndex];
        const actionValue = rank < 0 ? callPayoff[nodeIndex] : values[stateIndex(rank, 1 - node.player)];
        if (node.player === 0) {
          for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
            const probability = sigma0[nodeIndex][h0 * node.actions.length + actionIndex];
            for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
              const cell = h0 * HAND_COUNT + h1;
              value[cell] += probability * actionValue[cell];
            }
          }
        } else {
          for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
            const probability = sigma1[nodeIndex][h1 * node.actions.length + actionIndex];
            for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
              const cell = h0 * HAND_COUNT + h1;
              value[cell] += probability * actionValue[cell];
            }
          }
        }
      }
    }

    for (let nodeIndex = 0; nodeIndex < actions.length; nodeIndex += 1) {
      const node = actions[nodeIndex];
      if (node.player === 0) {
        for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
          let ownReach = 0;
          for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
            ownReach += handProbabilities[h1] * reach0[nodeIndex][h0 * HAND_COUNT + h1];
          }
          ownReachSum0[nodeIndex][h0] += ownReach;
          for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
            strategySum0[nodeIndex][h0 * node.actions.length + actionIndex] += ownReach * sigma0[nodeIndex][h0 * node.actions.length + actionIndex];
          }
        }
        for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
          const offset = h0 * node.actions.length;
          for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
            const cell = h0 * HAND_COUNT + h1;
            const counterfactualWeight = handProbabilities[h1] * reach1[nodeIndex][cell];
            for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
              const rank = node.actions[actionIndex];
              const actionValue = rank < 0 ? callPayoff[nodeIndex][cell] : values[stateIndex(rank, 1 - node.player)][cell];
              regret0[nodeIndex][offset + actionIndex] += counterfactualWeight * (actionValue - values[nodeIndex][cell]);
            }
          }
        }
      } else {
        for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
          let ownReach = 0;
          for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
            ownReach += handProbabilities[h0] * reach1[nodeIndex][h0 * HAND_COUNT + h1];
          }
          ownReachSum1[nodeIndex][h1] += ownReach;
          for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
            strategySum1[nodeIndex][h1 * node.actions.length + actionIndex] += ownReach * sigma1[nodeIndex][h1 * node.actions.length + actionIndex];
          }
        }
        for (let h1 = 0; h1 < HAND_COUNT; h1 += 1) {
          const offset = h1 * node.actions.length;
          for (let h0 = 0; h0 < HAND_COUNT; h0 += 1) {
            const cell = h0 * HAND_COUNT + h1;
            const counterfactualWeight = handProbabilities[h0] * reach0[nodeIndex][cell];
            for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
              const rank = node.actions[actionIndex];
              const actionValue = rank < 0 ? callPayoff[nodeIndex][cell] : values[stateIndex(rank, 1 - node.player)][cell];
              regret1[nodeIndex][offset + actionIndex] += counterfactualWeight * (values[nodeIndex][cell] - actionValue);
            }
          }
        }
      }
    }

    for (let nodeIndex = 0; nodeIndex < actions.length; nodeIndex += 1) {
      const node = actions[nodeIndex];
      for (let handIndex = 0; handIndex < HAND_COUNT; handIndex += 1) {
        positiveRegretStrategy(regret0[nodeIndex], sigma0[nodeIndex], node.actions, handIndex * node.actions.length);
        positiveRegretStrategy(regret1[nodeIndex], sigma1[nodeIndex], node.actions, handIndex * node.actions.length);
      }
    }
  }

  function averagePolicy(strategySum, reachSum, node) {
    const policy = [];
    for (let handIndex = 0; handIndex < HAND_COUNT; handIndex += 1) {
      const denominator = reachSum[handIndex];
      const values = [];
      let total = 0;
      for (let actionIndex = 0; actionIndex < node.actions.length; actionIndex += 1) {
        const value = denominator > 1e-12 ? strategySum[handIndex * node.actions.length + actionIndex] / denominator : 1 / node.actions.length;
        values.push(value);
        total += value;
      }
      policy.push(values.map((value) => value / total));
    }
    return policy;
  }

  return {
    hands,
    actions,
    average0: actions.map((node, nodeIndex) => averagePolicy(strategySum0[nodeIndex], ownReachSum0[nodeIndex], node)),
    average1: actions.map((node, nodeIndex) => averagePolicy(strategySum1[nodeIndex], ownReachSum1[nodeIndex], node))
  };
}

const iterations = Number.parseInt(process.argv[2] || DEFAULT_ITERATIONS, 10);
  const trained = train(iterations);

  // Keep the generated artifact deterministic and easy to load from a file:// page.
const model = {
    version: 1,
    nDice: DICE,
    maxQty: MAX_QTY,
    iterations,
    hands: trained.hands,
    actions: trained.actions.map((node) => node.actions),
    average0: quantizePolicy(trained.average0),
  average1: quantizePolicy(trained.average1)
};
const output = "window.LiarDiceCfrModel = " + JSON.stringify(model) + ";\n";
if (process.argv[3]) require("fs").writeFileSync(process.argv[3], output, "utf8");
else process.stdout.write(output);
