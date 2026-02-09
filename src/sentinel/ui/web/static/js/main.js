/* SENTINEL Dashboard — Main Entry Point
 *
 * Architecture: Data arrival (WebSocket, 10 Hz) is decoupled from rendering
 * (requestAnimationFrame, ~30-60 fps).  The tactical map renders at full
 * frame rate with position interpolation.  DOM panels are throttled per
 * MIL-STD-1472: readable values at 1 Hz, rate-of-change at 3-4 Hz.
 */

(function () {
  "use strict";

  var ws = null;
  var reconnectDelay = 1000;
  var maxReconnect = 10000;
  var statusEl = document.getElementById("ws-status");
  var timeEl = document.getElementById("header-time");

  // Data buffers for interpolation
  var currentData = null;
  var prevData = null;
  var lastDataTime = 0;
  var dataInterval = 100; // ms between data updates (recalculated from server config)

  // FPS tracking
  var frameCount = 0;
  var fpsStartTime = performance.now();
  var measuredFps = 0;

  // Throttle timestamps (ms)
  var lastTableUpdate = 0;
  var lastThreatUpdate = 0;
  var lastSensorUpdate = 0;
  var lastMetricsUpdate = 0;

  // Render cap: 30 fps target for tactical displays
  var RENDER_INTERVAL = 1000 / 30; // ~33.3ms
  var lastRenderTime = 0;

  // Throttle intervals per MIL-STD-1472
  var TABLE_INTERVAL = 250;    // 4 Hz — rate-of-change values
  var THREAT_INTERVAL = 500;   // 2 Hz — status panels
  var SENSOR_INTERVAL = 1000;  // 1 Hz — readable values
  var METRICS_INTERVAL = 500;  // 2 Hz — performance metrics

  // Clock
  setInterval(function () {
    var now = new Date();
    timeEl.textContent =
      String(now.getHours()).padStart(2, "0") + ":" +
      String(now.getMinutes()).padStart(2, "0") + ":" +
      String(now.getSeconds()).padStart(2, "0");
  }, 1000);

  function connect() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    var url = proto + "//" + location.host + "/ws/tracks";
    ws = new WebSocket(url);

    ws.onopen = function () {
      statusEl.textContent = "ONLINE";
      statusEl.className = "ws-indicator connected";
      reconnectDelay = 1000;
      // Fetch server config for data interval
      fetch("/api/config").then(function (r) { return r.json(); }).then(function (cfg) {
        if (cfg.track_update_hz > 0) dataInterval = 1000.0 / cfg.track_update_hz;
      }).catch(function () {});
    };

    ws.onmessage = function (event) {
      try {
        var data = JSON.parse(event.data);
        if (data.type === "update") {
          prevData = currentData;
          currentData = data;
          lastDataTime = performance.now();
        }
      } catch (e) {
        console.error("Parse error:", e);
      }
    };

    ws.onclose = function () {
      statusEl.textContent = "OFFLINE";
      statusEl.className = "ws-indicator disconnected";
      setTimeout(function () {
        reconnectDelay = Math.min(reconnectDelay * 1.5, maxReconnect);
        connect();
      }, reconnectDelay);
    };

    ws.onerror = function () {
      ws.close();
    };
  }

  // ---------------------------------------------------------------
  // Render loop — capped at 30 fps (tactical display standard)
  // ---------------------------------------------------------------
  function renderLoop() {
    requestAnimationFrame(renderLoop);

    if (!currentData) return;

    var now = performance.now();

    // Cap at 30 fps — skip frame if too soon
    if (now - lastRenderTime < RENDER_INTERVAL) return;
    lastRenderTime = now;

    // Interpolation factor: 0 = at prevData, 1 = at currentData
    var t = (dataInterval > 0) ? Math.min((now - lastDataTime) / dataInterval, 1.5) : 1.0;

    // FPS counter (1-second window)
    frameCount++;
    var fpsDelta = now - fpsStartTime;
    if (fpsDelta >= 1000) {
      measuredFps = Math.round(frameCount * 1000 / fpsDelta);
      frameCount = 0;
      fpsStartTime = now;
    }

    // Tactical map — full frame rate with interpolation
    try {
      window.TacticalMap.update(currentData, prevData, t);
    } catch (e) {
      console.error("TacticalMap:", e);
    }

    // Track table — 4 Hz (MIL-STD-1472 rate-of-change)
    if (now - lastTableUpdate >= TABLE_INTERVAL) {
      lastTableUpdate = now;
      try { window.TrackTable.update(currentData); } catch (e) { console.error("TrackTable:", e); }
    }

    // Threat panel — 2 Hz
    if (now - lastThreatUpdate >= THREAT_INTERVAL) {
      lastThreatUpdate = now;
      try { window.ThreatPanel.update(currentData); } catch (e) { console.error("ThreatPanel:", e); }
    }

    // Sensor panel — 1 Hz
    if (now - lastSensorUpdate >= SENSOR_INTERVAL) {
      lastSensorUpdate = now;
      try { window.SensorPanel.update(currentData.status || {}); } catch (e) { console.error("SensorPanel:", e); }
    }

    // Metrics panel — 2 Hz, inject measured render FPS
    if (now - lastMetricsUpdate >= METRICS_INTERVAL) {
      lastMetricsUpdate = now;
      try {
        var status = Object.assign({}, currentData.status || {});
        status.fps = measuredFps;
        window.MetricsPanel.update(status);
      } catch (e) { console.error("MetricsPanel:", e); }
    }
  }

  // Init
  document.addEventListener("DOMContentLoaded", function () {
    window.HudFeed.init();
    window.TacticalMap.resize();
    window.addEventListener("resize", function () { window.TacticalMap.resize(); });
    connect();
    requestAnimationFrame(renderLoop);
  });
})();
