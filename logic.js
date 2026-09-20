(function () {
  "use strict";

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function rollDice(count) {
    return Array.from({ length: count }, function () {
      return Math.floor(Math.random() * 6) + 1;
    });
  }

  function countMatching(dice, face) {
    return dice.reduce(function (total, die) {
      return total + (face === 1 ? (die === 1 ? 1 : 0) : (die === face || die === 1 ? 1 : 0));
    }, 0);
  }

  function matchProbability(face) {
    return face === 1 ? 1 / 6 : 1 / 3;
  }

  function combinations(n, k) {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    k = Math.min(k, n - k);
    var result = 1;
    for (var i = 1; i <= k; i += 1) {
      result = (result * (n - k + i)) / i;
    }
    return result;
  }

  function probabilityAtLeast(needed, diceCount, probability) {
    if (needed <= 0) return 1;
    if (needed > diceCount) return 0;
    var result = 0;
    for (var successes = needed; successes <= diceCount; successes += 1) {
      result += combinations(diceCount, successes) * Math.pow(probability, successes) * Math.pow(1 - probability, diceCount - successes);
    }
    return clamp(result, 0, 1);
  }

  function bidTruthProbability(bid, ownDice, opponentDiceCount) {
    var knownMatches = countMatching(ownDice, bid.face);
    return probabilityAtLeast(bid.quantity - knownMatches, opponentDiceCount, matchProbability(bid.face));
  }

  function bidIsValid(bid, currentBid, totalDice) {
    if (!bid || bid.quantity < 1 || bid.quantity > totalDice || bid.face < 1 || bid.face > 6) return false;
    if (!currentBid) return true;
    return bid.quantity > currentBid.quantity || (bid.quantity === currentBid.quantity && bid.face > currentBid.face);
  }

  function allValidBids(currentBid, totalDice) {
    var bids = [];
    for (var quantity = 1; quantity <= totalDice; quantity += 1) {
      for (var face = 1; face <= 6; face += 1) {
        var bid = { quantity: quantity, face: face };
        if (bidIsValid(bid, currentBid, totalDice)) bids.push(bid);
      }
    }
    return bids;
  }

  function positionValue(ownDiceCount, opponentDiceCount) {
    return ownDiceCount - opponentDiceCount;
  }

  // A player with a lead can afford to reject marginal challenges; a player behind needs variance.
  function challengeThreshold(ownDiceCount, opponentDiceCount, matchContext) {
    var total = Math.max(1, ownDiceCount + opponentDiceCount);
    var threshold = 0.5 + (ownDiceCount - opponentDiceCount) * 0.1 / total;
    if (matchContext && Number.isInteger(matchContext.targetWins)) {
      var ownAtMatchPoint = matchContext.ownWins >= matchContext.targetWins - 1;
      var opponentAtMatchPoint = matchContext.opponentWins >= matchContext.targetWins - 1;
      if (ownAtMatchPoint && !opponentAtMatchPoint) threshold += 0.05;
      if (opponentAtMatchPoint && !ownAtMatchPoint) threshold -= 0.05;
    }
    return clamp(threshold, 0.38, 0.62);
  }

  function shouldChallenge(bid, ownDice, opponentDiceCount, matchContext) {
    var truthProbability = bidTruthProbability(bid, ownDice, opponentDiceCount);
    var falseProbability = 1 - truthProbability;
    return {
      truthProbability: truthProbability,
      falseProbability: falseProbability,
      threshold: challengeThreshold(ownDice.length, opponentDiceCount, matchContext),
      challenge: falseProbability >= challengeThreshold(ownDice.length, opponentDiceCount, matchContext)
    };
  }

  function possibleOwnMatchCounts(diceCount, face) {
    var distribution = [];
    var probability = matchProbability(face);
    for (var matches = 0; matches <= diceCount; matches += 1) {
      distribution.push({
        matches: matches,
        probability: combinations(diceCount, matches) * Math.pow(probability, matches) * Math.pow(1 - probability, diceCount - matches)
      });
    }
    return distribution;
  }

  // Estimates how often a rational opponent would open this bid, without using their actual dice.
  function modeledChallengeProbability(bid, aiDiceCount, opponentDiceCount, matchContext) {
    var opponentContext = matchContext ? {
      ownWins: matchContext.opponentWins,
      opponentWins: matchContext.ownWins,
      targetWins: matchContext.targetWins
    } : null;
    var threshold = challengeThreshold(opponentDiceCount, aiDiceCount, opponentContext);
    return possibleOwnMatchCounts(opponentDiceCount, bid.face).reduce(function (result, item) {
      var opponentTruth = probabilityAtLeast(bid.quantity - item.matches, aiDiceCount, matchProbability(bid.face));
      return result + (1 - opponentTruth >= threshold ? item.probability : 0);
    }, 0);
  }

  function handKey(dice) {
    return dice.slice().sort(function (left, right) { return left - right; }).join(",");
  }

  function sampleWeightedAction(actions, weights) {
    var total = weights.reduce(function (sum, weight) { return sum + weight; }, 0);
    if (total <= 0) return actions[0];
    var cursor = Math.random() * total;
    for (var index = 0; index < actions.length; index += 1) {
      cursor -= weights[index];
      if (cursor <= 0) return actions[index];
    }
    return actions[actions.length - 1];
  }

  function cfrAction(aiDice, opponentDiceCount, currentBid, matchContext, aiPlayer) {
    var model = window.LiarDiceCfrModel;
    if (!model || model.nDice !== aiDice.length || model.nDice !== opponentDiceCount) return null;

    var handIndex = model.hands.findIndex(function (hand) { return hand.join(",") === handKey(aiDice); });
    if (handIndex < 0) return null;
    var rank = currentBid ? (currentBid.quantity - 1) * 6 + currentBid.face - 1 : -1;
    var player = Number.isInteger(aiPlayer) ? aiPlayer : 1;
    var nodeIndex = rank < 0 ? 0 : 1 + rank * 2 + player;
    var actions = model.actions[nodeIndex];
    var policyTable = player === 0 ? model.average0 : model.average1;
    var policy = policyTable[nodeIndex][handIndex].map(function (value) { return value / 1000000; });
    if (!actions || !policy || actions.length !== policy.length) return null;

    var challengeAnalysis = currentBid ? shouldChallenge(currentBid, aiDice, opponentDiceCount, matchContext) : null;
    var adjustedWeights = policy.slice();
    for (var index = 0; index < actions.length; index += 1) {
      if (actions[index] === -1 && challengeAnalysis) {
        var challengeBias = challengeAnalysis.challenge ? 1.65 : 0.62;
        adjustedWeights[index] *= challengeBias;
      }
    }
    var selectedAction = sampleWeightedAction(actions, adjustedWeights);
    if (selectedAction === -1) {
      return { type: "challenge", analysis: challengeAnalysis || { falseProbability: 0, truthProbability: 1 } };
    }
    var bid = { quantity: Math.floor(selectedAction / 6) + 1, face: selectedAction % 6 + 1 };
    return {
      type: "bid",
      bid: bid,
      analysis: {
        bid: bid,
        truthProbability: bidTruthProbability(bid, aiDice, opponentDiceCount),
        challengeProbability: modeledChallengeProbability(bid, aiDice.length, opponentDiceCount, matchContext),
        source: "cfr"
      }
    };
  }

  function evaluateRaise(bid, aiDice, opponentDiceCount, totalDice, matchContext) {
    var truthProbability = bidTruthProbability(bid, aiDice, opponentDiceCount);
    var challengeProbability = modeledChallengeProbability(bid, aiDice.length, opponentDiceCount, matchContext);
    var knownMatches = countMatching(aiDice, bid.face);
    var expectedMatches = knownMatches + opponentDiceCount * matchProbability(bid.face);
    var pressure = clamp((bid.quantity - expectedMatches) / Math.max(1, totalDice), -1, 1);
    var directChallengeDelta = 2 * truthProbability - 1;

    // A raise is valuable when it remains credible while pushing the opponent toward a mistake.
    var score = challengeProbability * directChallengeDelta;
    score += (1 - challengeProbability) * (truthProbability - 0.5) * 0.16;
    score += Math.max(0, pressure) * 0.08;
    score += Math.min(0.03, bid.quantity / Math.max(1, totalDice) * 0.03);

    return {
      bid: bid,
      score: score,
      truthProbability: truthProbability,
      challengeProbability: challengeProbability,
      pressure: pressure
    };
  }

  function chooseAiAction(aiDice, opponentDiceCount, currentBid, matchContext, aiPlayer) {
    var totalDice = aiDice.length + opponentDiceCount;
    var modelAction = cfrAction(aiDice, opponentDiceCount, currentBid, matchContext, aiPlayer);
    if (modelAction && modelAction.type === "bid" && !bidIsValid(modelAction.bid, currentBid, totalDice)) modelAction = null;
    if (modelAction) return modelAction;
    if (currentBid) {
      var challengeAnalysis = shouldChallenge(currentBid, aiDice, opponentDiceCount, matchContext);
      if (challengeAnalysis.challenge || allValidBids(currentBid, totalDice).length === 0) {
        return { type: "challenge", analysis: challengeAnalysis };
      }
    }

    var candidates = allValidBids(currentBid, totalDice).map(function (bid) {
      return evaluateRaise(bid, aiDice, opponentDiceCount, totalDice, matchContext);
    });

    if (candidates.length === 0) {
      return { type: "challenge", analysis: shouldChallenge(currentBid, aiDice, opponentDiceCount, matchContext) };
    }

    candidates.sort(function (left, right) {
      return right.score - left.score;
    });

    // A tiny tie-break randomizer keeps the mixed strategy from becoming predictable.
    var bestScore = candidates[0].score;
    var finalists = candidates.filter(function (candidate) {
      return bestScore - candidate.score < 0.012;
    });
    var chosen = finalists[Math.floor(Math.random() * finalists.length)];
    return { type: "bid", bid: chosen.bid, analysis: chosen };
  }

  window.LiarDiceLogic = {
    rollDice: rollDice,
    countMatching: countMatching,
    bidTruthProbability: bidTruthProbability,
    bidIsValid: bidIsValid,
    allValidBids: allValidBids,
    probabilityAtLeast: probabilityAtLeast,
    chooseAiAction: chooseAiAction,
    shouldChallenge: shouldChallenge,
    modeledChallengeProbability: modeledChallengeProbability,
    cfrAction: cfrAction
  };
}());
