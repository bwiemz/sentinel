/* SENTINEL Performance Metrics Panel */

window.MetricsPanel = (function () {
  const fpsEl = document.getElementById("fps-value");
  const uptimeEl = document.getElementById("uptime-value");
  const countsEl = document.getElementById("track-counts");
  const barsEl = document.getElementById("latency-bars");
  let lastCountsHtml = "";
  let lastBarsHtml = "";

  const STAGES = [
    { key: "detect_ms", label: "DETECT" },
    { key: "track_ms", label: "TRACK" },
    { key: "radar_ms", label: "RADAR" },
    { key: "fusion_ms", label: "FUSION" },
    { key: "render_ms", label: "RENDER" },
  ];

  function formatUptime(seconds) {
    if (seconds == null) return "--:--:--";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return (
      String(h).padStart(2, "0") + ":" +
      String(m).padStart(2, "0") + ":" +
      String(s).padStart(2, "0")
    );
  }

  function update(status) {
    // FPS — this is now the actual render FPS (measured client-side)
    const fps = status.fps;
    fpsEl.textContent = fps != null ? String(fps) : "--";
    fpsEl.style.color = fps != null && fps < 15 ? "var(--color-danger)" : "var(--color-primary)";

    // Uptime
    uptimeEl.textContent = formatUptime(status.uptime);

    // Track counts
    let countsHtml = "";
    if (status.track_count != null) countsHtml += "CAM: " + (status.confirmed_count || 0) + "/" + status.track_count + "<br>";
    if (status.radar_track_count != null) countsHtml += "RDR: " + status.radar_track_count + "<br>";
    if (status.thermal_track_count != null) countsHtml += "THM: " + status.thermal_track_count + "<br>";
    if (status.fused_track_count != null) countsHtml += "FUS: " + status.fused_track_count + "<br>";
    // System info (Phase 14-20)
    if (status.clock_mode) {
      countsHtml += "CLK: " + status.clock_mode.toUpperCase();
      if (status.sim_time != null) countsHtml += " T=" + status.sim_time.toFixed(1);
      countsHtml += "<br>";
    }
    if (status.cpp_accel) {
      var accelParts = [];
      if (status.cpp_accel.core) accelParts.push("C++");
      if (status.cpp_accel.batch) accelParts.push("BATCH");
      countsHtml += "ACCEL: " + (accelParts.length > 0 ? accelParts.join("+") : "PY") + "<br>";
    }
    var featureFlags = [];
    if (status.iff_enabled) featureFlags.push("IFF");
    if (status.roe_enabled) featureFlags.push("ROE");
    if (status.ew_enabled) featureFlags.push("EW");
    if (status.network_enabled) featureFlags.push("NET:" + (status.network_node_id || "LOCAL"));
    if (featureFlags.length > 0) countsHtml += featureFlags.join(" | ");
    if (countsHtml !== lastCountsHtml) {
      countsEl.innerHTML = countsHtml;
      lastCountsHtml = countsHtml;
    }

    // Latency bars
    const maxMs = 40; // scale: 40ms = full bar
    let barsHtml = "";
    for (const stage of STAGES) {
      const val = status[stage.key];
      if (val == null) continue;
      const pct = Math.min(100, (val / maxMs) * 100);
      const color = val > 30 ? "var(--color-danger)" : val > 15 ? "var(--color-medium)" : "var(--color-primary)";
      barsHtml +=
        '<div class="latency-row">' +
        '<span class="latency-label">' + stage.label + "</span>" +
        '<div class="latency-bar-bg"><div class="latency-bar-fill" style="width:' +
        pct + "%;background:" + color + '"></div></div>' +
        '<span class="latency-value">' + val.toFixed(1) + "ms</span></div>";
    }
    if (barsHtml !== lastBarsHtml) {
      barsEl.innerHTML = barsHtml;
      lastBarsHtml = barsHtml;
    }
  }

  return { update: update };
})();
