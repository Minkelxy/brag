(function () {
  "use strict";

  var logic = window.LiarDiceLogic;
  var GAME_RULES = {
    dicePerPlayer: 3,
    defaultMatchRounds: 3
  };
  var els = {
    roundLabel: document.getElementById("roundLabel"),
    turnLabel: document.getElementById("turnLabel"),
    humanWins: document.getElementById("humanWins"),
    aiWins: document.getElementById("aiWins"),
    matchLengthInput: document.getElementById("matchLengthInput"),
    matchFormatLabel: document.getElementById("matchFormatLabel"),
    humanDiceTray: document.getElementById("humanDiceTray"),
    aiDiceTray: document.getElementById("aiDiceTray"),
    bidDisplay: document.getElementById("bidDisplay"),
    bidHint: document.getElementById("bidHint"),
    moveStatus: document.getElementById("moveStatus"),
    quantityInput: document.getElementById("quantityInput"),
    faceInput: document.getElementById("faceInput"),
    bidButton: document.getElementById("bidButton"),
    challengeButton: document.getElementById("challengeButton"),
    validationMessage: document.getElementById("validationMessage"),
    eventLog: document.getElementById("eventLog"),
    newGameButton: document.getElementById("newGameButton"),
    resultModal: document.getElementById("resultModal"),
    resultTitle: document.getElementById("resultTitle"),
    resultSummary: document.getElementById("resultSummary"),
    resultHumanDice: document.getElementById("resultHumanDice"),
    resultAiDice: document.getElementById("resultAiDice"),
    resultScore: document.getElementById("resultScore"),
    resultDetail: document.getElementById("resultDetail"),
    continueButton: document.getElementById("continueButton")
  };

  var state = null;
  var aiTimer = null;

  function targetWinsFor(matchRounds) {
    return Math.ceil(matchRounds / 2);
  }

  function matchFormatText(matchRounds) {
    var targetWins = targetWinsFor(matchRounds);
    return matchRounds === 1 ? "单局 · 双方固定 3 颗骰子" : matchRounds + " 局 " + targetWins + " 胜 · 双方固定 3 颗骰子";
  }

  function createState(matchRounds) {
    var totalRounds = Number.isInteger(matchRounds) ? matchRounds : GAME_RULES.defaultMatchRounds;
    return {
      round: 1,
      matchRounds: totalRounds,
      targetWins: targetWinsFor(totalRounds),
      humanWins: 0,
      aiWins: 0,
      humanDice: logic.rollDice(GAME_RULES.dicePerPlayer),
      aiDice: logic.rollDice(GAME_RULES.dicePerPlayer),
      currentPlayer: "human",
      bid: null,
      eventLog: [],
      phase: "playing",
      starter: "human",
      matchOver: false
    };
  }

  function dieElement(value, hidden) {
    var element = document.createElement("span");
    element.className = "die" + (hidden ? " hidden-die" : "");
    element.textContent = hidden ? "?" : String(value);
    element.setAttribute("aria-label", hidden ? "隐藏骰子" : value + " 点");
    return element;
  }

  function renderDice(container, dice, hidden) {
    container.innerHTML = "";
    dice.forEach(function (value) {
      container.appendChild(dieElement(value, hidden));
    });
  }

  function faceName(face) {
    return ["", "一", "二", "三", "四", "五", "六"][face] + "点";
  }

  function bidText(bid) {
    return bid ? bid.quantity + " 个" + faceName(bid.face) : "等待第一口";
  }

  function addLog(message, important) {
    state.eventLog.push({ message: message, important: Boolean(important) });
    if (state.eventLog.length > 9) state.eventLog.shift();
  }

  function renderLog() {
    els.eventLog.innerHTML = "";
    state.eventLog.slice().reverse().forEach(function (entry) {
      var item = document.createElement("li");
      item.textContent = entry.message;
      if (entry.important) item.className = "important";
      els.eventLog.appendChild(item);
    });
  }

  function renderBid() {
    if (!state.bid) {
      els.bidDisplay.textContent = "等待第一口";
      els.bidDisplay.className = "bid-display empty";
      els.bidHint.textContent = "先叫一个数量和点数";
      return;
    }
    els.bidDisplay.textContent = bidText(state.bid);
    els.bidDisplay.className = "bid-display";
    els.bidHint.textContent = state.bid.by === "human" ? "这是你的叫法，AI 正在判断" : "你可以继续加注，或选择开盅";
  }

  function validHumanBid() {
    var quantity = Number.parseInt(els.quantityInput.value, 10);
    var face = Number.parseInt(els.faceInput.value, 10);
    var bid = { quantity: quantity, face: face };
    return {
      bid: bid,
      valid: Number.isInteger(quantity) && Number.isInteger(face) && logic.bidIsValid(bid, state.bid, state.humanDice.length + state.aiDice.length)
    };
  }

  function renderControls() {
    var isHumanTurn = state.phase === "playing" && state.currentPlayer === "human";
    var validation = validHumanBid();
    els.bidButton.disabled = !isHumanTurn || !validation.valid;
    els.challengeButton.disabled = !isHumanTurn || !state.bid || state.bid.by !== "ai";
    els.quantityInput.disabled = !isHumanTurn;
    els.faceInput.disabled = !isHumanTurn;

    if (!isHumanTurn) {
      els.moveStatus.textContent = state.phase === "playing" ? "AI 思考中" : "本局结束";
      els.validationMessage.textContent = "等对手完成动作。";
      els.validationMessage.className = "validation-message";
      return;
    }

    els.moveStatus.textContent = "轮到你";
    if (!validation.valid) {
      els.validationMessage.textContent = state.bid ? "加注必须高于当前叫法。" : "请输入有效的数量和点数。";
      els.validationMessage.className = "validation-message invalid";
    } else {
      els.validationMessage.textContent = "叫得越高，风险越大；也更容易逼对手开盅。";
      els.validationMessage.className = "validation-message";
    }
  }

  function render() {
    els.roundLabel.textContent = "第 " + state.round + " / " + state.matchRounds + " 局";
    els.turnLabel.textContent = state.phase === "playing" ? (state.currentPlayer === "human" ? "你的回合" : "AI 回合") : (state.matchOver ? "比赛结束" : "等待下一局");
    els.humanWins.textContent = String(state.humanWins);
    els.aiWins.textContent = String(state.aiWins);
    els.matchFormatLabel.textContent = matchFormatText(state.matchRounds);
    renderDice(els.humanDiceTray, state.humanDice, false);
    renderDice(els.aiDiceTray, state.aiDice, true);
    renderBid();
    renderControls();
    renderLog();
  }

  function setBid(bid, by) {
    state.bid = { quantity: bid.quantity, face: bid.face, by: by };
    addLog((by === "human" ? "你" : "AI") + "叫了 " + bidText(bid) + "。", by === "human");
  }

  function submitHumanBid() {
    if (state.phase !== "playing" || state.currentPlayer !== "human") return;
    var validation = validHumanBid();
    if (!validation.valid) return;
    setBid(validation.bid, "human");
    state.currentPlayer = "ai";
    render();
    scheduleAiTurn();
  }

  function scheduleAiTurn() {
    window.clearTimeout(aiTimer);
    aiTimer = window.setTimeout(playAiTurn, 520);
  }

  function aiReason(action) {
    if (action.type === "challenge") {
      var falsePercent = Math.round(action.analysis.falseProbability * 100);
      return "AI 判断当前叫法为假的概率约 " + falsePercent + "%。";
    }
    var confidence = Math.round(action.analysis.truthProbability * 100);
    var pressure = action.analysis.challengeProbability >= 0.45 ? "提高了你开盅的诱因" : "保留了继续加注的空间";
    var source = action.analysis.source === "cfr" ? "CFR 策略选择了" : "概率策略选择了";
    return source + "一个约 " + confidence + "% 可信、同时" + pressure + "的叫法。";
  }

  function playAiTurn() {
    if (state.phase !== "playing" || state.currentPlayer !== "ai") return;
    var action = logic.chooseAiAction(state.aiDice, state.humanDice.length, state.bid, {
      ownWins: state.aiWins,
      opponentWins: state.humanWins,
      targetWins: state.targetWins
    }, state.starter === "ai" ? 0 : 1);
    if (action.type === "challenge") {
      addLog("AI 选择开盅。", true);
      render();
      resolveRound("ai", action.analysis);
      return;
    }
    setBid(action.bid, "ai");
    addLog(aiReason(action));
    state.currentPlayer = "human";
    render();
  }

  function submitHumanChallenge() {
    if (state.phase !== "playing" || state.currentPlayer !== "human" || !state.bid || state.bid.by !== "ai") return;
    addLog("你选择开盅。", true);
    render();
    resolveRound("human");
  }

  function totalMatches(bid) {
    return logic.countMatching(state.humanDice, bid.face) + logic.countMatching(state.aiDice, bid.face);
  }

  function resolveRound(challenger, analysis) {
    state.phase = "resolving";
    render();
    var total = totalMatches(state.bid);
    var bidIsTrue = total >= state.bid.quantity;
    var winner = bidIsTrue ? state.bid.by : challenger;
    if (winner === "human") state.humanWins += 1;
    else state.aiWins += 1;
    state.matchOver = state.humanWins >= state.targetWins || state.aiWins >= state.targetWins;

    var winnerLabel = winner === "human" ? "你" : "AI";
    var resultTitle = winnerLabel + "赢下这一局";
    var resultSummary = bidText(state.bid) + "，实际数到 " + total + " 个。" + (bidIsTrue ? "叫法成立，叫牌者获胜。" : "叫法不成立，开盅者获胜。");
    if (state.matchOver) resultTitle = winnerLabel === "你" ? "你赢下整场比赛" : "AI 赢下整场比赛";

    els.resultTitle.textContent = resultTitle;
    els.resultSummary.textContent = resultSummary;
    renderDice(els.resultHumanDice, state.humanDice, false);
    renderDice(els.resultAiDice, state.aiDice, false);
    els.resultScore.textContent = "当前比分：你 " + state.humanWins + " : " + state.aiWins + " AI";
    els.resultDetail.textContent = analysis && challenger === "ai" ? "AI 的开盅判断已按精确概率完成。" : (state.matchOver ? "比赛达到目标胜局。" : "下一局会重新摇出双方各 3 颗骰子。");
    els.continueButton.textContent = state.matchOver ? "再来一场" : "下一局";
    els.resultModal.classList.remove("hidden");
    state.phase = "roundEnd";
    render();
  }

  function continueGame() {
    els.resultModal.classList.add("hidden");
    if (state.matchOver) {
      startNewGame();
      return;
    }
    state.round += 1;
    state.humanDice = logic.rollDice(GAME_RULES.dicePerPlayer);
    state.aiDice = logic.rollDice(GAME_RULES.dicePerPlayer);
    state.bid = null;
    state.eventLog = [];
    state.phase = "playing";
    state.starter = state.round % 2 === 0 ? "ai" : "human";
    state.currentPlayer = state.starter;
    addLog("第 " + state.round + " 局开始，" + (state.starter === "human" ? "你" : "AI") + "先叫。", true);
    render();
    if (state.currentPlayer === "ai") scheduleAiTurn();
  }

  function startNewGame() {
    window.clearTimeout(aiTimer);
    var matchRounds = Number.parseInt(els.matchLengthInput.value, 10);
    state = createState(matchRounds);
    addLog("新比赛开始，共 " + state.matchRounds + " 局，目标 " + state.targetWins + " 胜。你先叫。", true);
    els.resultModal.classList.add("hidden");
    render();
  }

  els.bidButton.addEventListener("click", submitHumanBid);
  els.challengeButton.addEventListener("click", submitHumanChallenge);
  els.newGameButton.addEventListener("click", startNewGame);
  els.continueButton.addEventListener("click", continueGame);
  els.matchLengthInput.addEventListener("change", startNewGame);
  els.quantityInput.addEventListener("input", renderControls);
  els.faceInput.addEventListener("change", renderControls);

  startNewGame();
}());
