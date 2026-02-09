/* SENTINEL Dashboard — Main Entry Point */

(function () {
  "use strict";

  var ws = null;
  var reconnectDelay = 1000;
  var maxReconnect = 10000;
  var statusEl = document.getElementById("ws-status");
  var timeEl = document.getElementById("header-time");
  var pendingData = null;
  var rafScheduled = false;

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
    };

    ws.onmessage = function (event) {
      try {
        var data = JSON.parse(event.data);
        if (data.type === "update") {
          pendingData = data;
          if (!rafScheduled) {
            rafScheduled = true;
            requestAnimationFrame(function () {
              rafScheduled = false;
              if (pendingData) {
                dispatch(pendingData);
                pendingData = null;
              }
            });
          }
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

  function dispatch(data) {
    try { window.TacticalMap.update(data); } catch (e) { console.error("TacticalMap:", e); }
    try { window.SensorPanel.update(data.status || {}); } catch (e) { console.error("SensorPanel:", e); }
    try { window.TrackTable.update(data); } catch (e) { console.error("TrackTable:", e); }
    try { window.ThreatPanel.update(data); } catch (e) { console.error("ThreatPanel:", e); }
    try { window.MetricsPanel.update(data.status || {}); } catch (e) { console.error("MetricsPanel:", e); }
  }

  // Init
  document.addEventListener("DOMContentLoaded", function () {
    window.HudFeed.init();
    window.TacticalMap.resize();
    window.addEventListener("resize", function () { window.TacticalMap.resize(); });
    connect();
  });
})();
