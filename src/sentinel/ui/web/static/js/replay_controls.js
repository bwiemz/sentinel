/* SENTINEL Dashboard — Replay Controls Module
 *
 * Recording (start/stop/pause/export) and Playback (play/pause/stop/seek/speed)
 * controls for the Phase 22 Track History & Replay system.
 *
 * Exposed as window.ReplayControls.update(status) — called from main.js at 1 Hz.
 */

(function () {
  "use strict";

  var panel = document.getElementById("replay-panel");
  var modeLabel = document.getElementById("replay-mode-label");

  // Recording elements
  var recIndicator = document.getElementById("rec-indicator");
  var recFrameCount = document.getElementById("rec-frame-count");
  var btnRecStart = document.getElementById("btn-rec-start");
  var btnRecStop = document.getElementById("btn-rec-stop");
  var btnRecPause = document.getElementById("btn-rec-pause");
  var btnExport = document.getElementById("btn-export");

  // Playback elements
  var btnStepBack = document.getElementById("btn-step-back");
  var btnPlay = document.getElementById("btn-play");
  var btnPause = document.getElementById("btn-pause");
  var btnStop = document.getElementById("btn-stop-replay");
  var btnStepFwd = document.getElementById("btn-step-fwd");
  var speedSelect = document.getElementById("replay-speed");

  // Timeline elements
  var timelineCurrent = document.getElementById("timeline-current");
  var timelineSlider = document.getElementById("timeline-slider");
  var timelineTotal = document.getElementById("timeline-total");
  var timelineFrame = document.getElementById("timeline-frame");

  var lastRecState = null;
  var lastPlayState = null;
  var sliderDragging = false;

  // ---------------------------------------------------------------
  // API helpers
  // ---------------------------------------------------------------

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (r) { return r.json(); }).catch(function (e) {
      console.error("ReplayControls POST " + url, e);
    });
  }

  // ---------------------------------------------------------------
  // Recording controls
  // ---------------------------------------------------------------

  if (btnRecStart) btnRecStart.addEventListener("click", function () { post("/api/history/start"); });
  if (btnRecStop) btnRecStop.addEventListener("click", function () { post("/api/history/stop"); });
  if (btnRecPause) btnRecPause.addEventListener("click", function () { post("/api/history/pause"); });

  if (btnExport) btnExport.addEventListener("click", function () {
    post("/api/history/export").then(function (resp) {
      if (resp && resp.path) {
        btnExport.textContent = "SAVED";
        setTimeout(function () { btnExport.textContent = "EXPORT"; }, 2000);
      }
    });
  });

  // ---------------------------------------------------------------
  // Playback controls
  // ---------------------------------------------------------------

  if (btnPlay) btnPlay.addEventListener("click", function () { post("/api/replay/play"); });
  if (btnPause) btnPause.addEventListener("click", function () { post("/api/replay/pause"); });
  if (btnStop) btnStop.addEventListener("click", function () { post("/api/replay/stop"); });
  if (btnStepBack) btnStepBack.addEventListener("click", function () { post("/api/replay/seek", { frame: "__step_back__" }); });
  if (btnStepFwd) btnStepFwd.addEventListener("click", function () { post("/api/replay/seek", { frame: "__step_fwd__" }); });

  if (speedSelect) speedSelect.addEventListener("change", function () {
    post("/api/replay/speed", { speed: parseFloat(speedSelect.value) });
  });

  // Timeline slider
  if (timelineSlider) {
    timelineSlider.addEventListener("mousedown", function () { sliderDragging = true; });
    timelineSlider.addEventListener("touchstart", function () { sliderDragging = true; });
    timelineSlider.addEventListener("change", function () {
      sliderDragging = false;
      post("/api/replay/seek", { frame: parseInt(timelineSlider.value, 10) });
    });
    timelineSlider.addEventListener("mouseup", function () { sliderDragging = false; });
    timelineSlider.addEventListener("touchend", function () { sliderDragging = false; });
  }

  // ---------------------------------------------------------------
  // Format helpers
  // ---------------------------------------------------------------

  function fmtTime(seconds) {
    if (seconds == null || isNaN(seconds)) return "--:--";
    var s = Math.floor(seconds);
    var m = Math.floor(s / 60);
    s = s % 60;
    return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
  }

  // ---------------------------------------------------------------
  // Update from main render loop
  // ---------------------------------------------------------------

  function update(status) {
    if (!panel) return;

    var historyEnabled = status.history_enabled;

    // Hide entire panel if history not enabled
    if (!historyEnabled) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "";

    var hist = status.history || {};
    var recState = hist.state || "idle";
    var replayMode = status.replay_mode || false;
    var replayState = status.replay_state || "stopped";
    var replayIndex = status.replay_index || 0;
    var replayTotal = status.replay_total || 0;
    var replaySpeed = status.replay_speed || 1.0;
    var replayFrame = status.replay_frame || null;

    // Mode label
    if (modeLabel) {
      if (replayMode) {
        modeLabel.textContent = "REPLAY";
        modeLabel.className = "mode-label replay";
      } else {
        modeLabel.textContent = "LIVE";
        modeLabel.className = "mode-label live";
      }
    }

    // Recording indicator
    if (recIndicator) {
      recIndicator.className = "rec-dot " + recState;
      recIndicator.title = recState.toUpperCase();
    }

    // Frame counter
    if (recFrameCount) {
      recFrameCount.textContent = (hist.recorded_count || 0) + " frames";
    }

    // Recording button states
    if (btnRecStart) btnRecStart.disabled = (recState === "recording");
    if (btnRecStop) btnRecStop.disabled = (recState === "idle");
    if (btnRecPause) btnRecPause.disabled = (recState !== "recording");
    if (btnExport) btnExport.disabled = (hist.recorded_count || 0) === 0;

    // Playback button states
    var hasData = replayTotal > 0 || (hist.buffer_frames || 0) > 0;
    if (btnPlay) btnPlay.disabled = !hasData || replayState === "playing";
    if (btnPause) btnPause.disabled = replayState !== "playing";
    if (btnStop) btnStop.disabled = replayState === "stopped" && !replayMode;
    if (btnStepBack) btnStepBack.disabled = !hasData;
    if (btnStepFwd) btnStepFwd.disabled = !hasData;

    // Speed selector
    if (speedSelect && parseFloat(speedSelect.value) !== replaySpeed) {
      speedSelect.value = String(replaySpeed);
    }

    // Timeline
    var total = replayTotal || hist.buffer_frames || 0;
    if (timelineSlider && !sliderDragging) {
      timelineSlider.max = Math.max(total - 1, 0);
      timelineSlider.value = replayIndex;
    }

    if (timelineFrame) {
      timelineFrame.textContent = replayIndex + " / " + total;
    }

    // Time display from replay frame data
    if (replayFrame && timelineCurrent) {
      timelineCurrent.textContent = fmtTime(replayFrame.elapsed || 0);
    }
    if (hist.time_range && timelineTotal) {
      var range = hist.time_range;
      timelineTotal.textContent = fmtTime(range[1] - range[0]);
    }
  }

  window.ReplayControls = { update: update };
})();
