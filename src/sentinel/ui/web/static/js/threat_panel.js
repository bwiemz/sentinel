/* SENTINEL Threat Panel — incremental DOM updates */

window.ThreatPanel = (function () {
  const cardsEl = document.getElementById("threat-cards");
  const detailsEl = document.getElementById("threat-details");

  const LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
  const CSS_CLASS = { CRITICAL: "critical", HIGH: "high", MEDIUM: "medium", LOW: "low" };
  const COLORS = {
    critical: "var(--color-critical)",
    high: "var(--color-high)",
  };

  // Pre-create the 4 card elements once
  var cardEls = [];
  for (var i = 0; i < LEVELS.length; i++) {
    var card = document.createElement("div");
    card.className = "threat-card " + CSS_CLASS[LEVELS[i]];
    var countDiv = document.createElement("div");
    countDiv.className = "count";
    countDiv.textContent = "0";
    var labelDiv = document.createElement("div");
    labelDiv.className = "label";
    labelDiv.textContent = LEVELS[i];
    card.appendChild(countDiv);
    card.appendChild(labelDiv);
    cardsEl.appendChild(card);
    cardEls.push(countDiv);
  }

  // Detail item pool
  var detailPool = [];
  var noThreatsEl = document.createElement("span");
  noThreatsEl.style.color = "var(--color-text-dim)";
  noThreatsEl.textContent = "No active threats";

  function ensureDetailItem(index) {
    if (index < detailPool.length) return detailPool[index];
    var item = document.createElement("div");
    item.className = "threat-detail-item";
    var badge = document.createElement("span");
    badge.className = "threat-badge";
    var textNode = document.createTextNode("");
    item.appendChild(badge);
    item.appendChild(textNode);
    detailPool.push(item);
    return item;
  }

  function update(data) {
    var status = data.status || {};
    var counts = status.threat_counts || {};
    var enhanced = (data.tracks || {}).enhanced_fused || [];

    // Update card counts in place
    for (var i = 0; i < LEVELS.length; i++) {
      var val = String(counts[LEVELS[i]] || 0);
      if (cardEls[i].textContent !== val) cardEls[i].textContent = val;
    }

    // Build list of CRITICAL/HIGH items
    var items = [];
    for (var j = 0; j < enhanced.length; j++) {
      var t = enhanced[j];
      if (t.threat_level === "CRITICAL" || t.threat_level === "HIGH") {
        items.push(t);
      }
    }

    // Remove "no threats" placeholder if present
    if (noThreatsEl.parentNode === detailsEl) {
      detailsEl.removeChild(noThreatsEl);
    }

    if (items.length === 0) {
      // Hide all detail items, show placeholder
      for (var k = 0; k < detailPool.length; k++) {
        if (detailPool[k].parentNode === detailsEl) {
          detailsEl.removeChild(detailPool[k]);
        }
      }
      detailsEl.appendChild(noThreatsEl);
      return;
    }

    // Update/create detail items
    for (var m = 0; m < items.length; m++) {
      var t = items[m];
      var el = ensureDetailItem(m);
      var cls = CSS_CLASS[t.threat_level] || "";
      var color = cls === "critical" ? COLORS.critical : COLORS.high;

      // Update badge color
      var badge = el.children[0];
      if (badge.style.background !== color) badge.style.background = color;

      // Update text — now includes intent and confidence (Phase 18)
      var text = t.threat_level + " | " +
        (t.fused_id || "").substring(0, 8) +
        (t.range_m != null ? " | R:" + (t.range_m / 1000).toFixed(1) + "km" : "") +
        (t.intent && t.intent !== "unknown" ? " | " + t.intent.toUpperCase() : "") +
        (t.threat_confidence ? " | " + Math.round(t.threat_confidence * 100) + "%" : "") +
        (t.is_stealth_candidate ? " | STEALTH" : "") +
        (t.is_hypersonic_candidate ? " | HYPERSONIC" : "");

      if (el.childNodes[1].textContent !== text) {
        el.childNodes[1].textContent = text;
      }

      // Ensure it's in the DOM
      if (el.parentNode !== detailsEl) {
        detailsEl.appendChild(el);
      }
    }

    // Remove excess detail items
    for (var n = items.length; n < detailPool.length; n++) {
      if (detailPool[n].parentNode === detailsEl) {
        detailsEl.removeChild(detailPool[n]);
      }
    }
  }

  return { update: update };
})();
