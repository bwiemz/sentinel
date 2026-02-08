/* SENTINEL Threat Panel */

window.ThreatPanel = (function () {
  const cardsEl = document.getElementById("threat-cards");
  const detailsEl = document.getElementById("threat-details");

  const LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
  const CSS_CLASS = { CRITICAL: "critical", HIGH: "high", MEDIUM: "medium", LOW: "low" };

  function update(data) {
    const status = data.status || {};
    const counts = status.threat_counts || {};
    const enhanced = (data.tracks || {}).enhanced_fused || [];

    // Cards
    let cardsHtml = "";
    for (const level of LEVELS) {
      const count = counts[level] || 0;
      cardsHtml +=
        '<div class="threat-card ' + CSS_CLASS[level] + '">' +
        '<div class="count">' + count + "</div>" +
        '<div class="label">' + level + "</div></div>";
    }
    cardsEl.innerHTML = cardsHtml;

    // Detail list for CRITICAL and HIGH
    let detHtml = "";
    for (const t of enhanced) {
      if (t.threat_level === "CRITICAL" || t.threat_level === "HIGH") {
        const cls = CSS_CLASS[t.threat_level] || "";
        const color = cls === "critical" ? "var(--color-critical)" : "var(--color-high)";
        detHtml +=
          '<div class="threat-detail-item">' +
          '<span class="threat-badge" style="background:' + color + '"></span>' +
          t.threat_level + " | " +
          (t.fused_id || "").substring(0, 8) +
          (t.range_m != null ? " | R:" + (t.range_m / 1000).toFixed(1) + "km" : "") +
          (t.is_stealth_candidate ? " | STEALTH" : "") +
          (t.is_hypersonic_candidate ? " | HYPERSONIC" : "") +
          "</div>";
      }
    }
    detailsEl.innerHTML = detHtml || '<span style="color:var(--color-text-dim)">No active threats</span>';
  }

  return { update: update };
})();
