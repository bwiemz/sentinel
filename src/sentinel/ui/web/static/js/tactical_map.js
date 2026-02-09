/* SENTINEL Tactical Map — PPI Radar Scope with interpolation */

window.TacticalMap = (function () {
  const canvas = document.getElementById("ppi-canvas");
  const ctx = canvas.getContext("2d");
  let maxRange = 20; // km

  const COLORS = {
    grid: "#003300",
    ring_text: "#006400",
    radar: "#00c8ff",
    thermal: "#ff6400",
    camera: "#00ff00",
    fused: "#ffffff",
    critical: "#ff0000",
    high: "#ff5000",
    medium: "#ffc800",
    low: "#00c800",
    confirmed: "#00ff00",
    tentative: "#ffff00",
    coasting: "#ffa500",
  };

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
  }

  // ---------------------------------------------------------------
  // Interpolation helpers
  // ---------------------------------------------------------------
  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function lerpAngle(a, b, t) {
    // Shortest-arc interpolation for degrees (-180 to 180)
    var diff = b - a;
    if (diff > 180) diff -= 360;
    if (diff < -180) diff += 360;
    return a + diff * t;
  }

  // Build lookup by track_id from a track array
  function buildLookup(arr) {
    var map = {};
    if (!arr) return map;
    for (var i = 0; i < arr.length; i++) {
      var key = arr[i].track_id || arr[i].fused_id;
      if (key) map[key] = arr[i];
    }
    return map;
  }

  // ---------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------
  function drawBackground() {
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;
    const radius = Math.min(cx, cy) - 10;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#010a01";
    ctx.fillRect(0, 0, w, h);

    // Range rings
    const rings = [0.25, 0.5, 0.75, 1.0];
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (const pct of rings) {
      const r = radius * pct;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = COLORS.ring_text;
      ctx.font = "9px Consolas, monospace";
      ctx.fillText((maxRange * pct).toFixed(0) + "km", cx + 4, cy - r + 10);
    }

    // Azimuth lines (every 30 degrees)
    for (let deg = 0; deg < 360; deg += 30) {
      const rad = (deg - 90) * (Math.PI / 180);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad) * radius, cy + Math.sin(rad) * radius);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 0.3;
      ctx.stroke();
    }

    // Center dot
    ctx.fillStyle = COLORS.confirmed;
    ctx.beginPath();
    ctx.arc(cx, cy, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function plotTrack(cx, cy, radius, range_m, azimuth_deg, type, state, threat, label) {
    if (range_m == null || azimuth_deg == null) return;
    const range_km = range_m / 1000;
    if (range_km > maxRange) return;

    const r = (range_km / maxRange) * radius;
    const rad = (azimuth_deg - 90) * (Math.PI / 180);
    const x = cx + Math.cos(rad) * r;
    const y = cy + Math.sin(rad) * r;

    let color = COLORS[state] || COLORS.confirmed;
    if (threat === "CRITICAL") color = COLORS.critical;
    else if (threat === "HIGH") color = COLORS.high;
    else if (threat === "MEDIUM") color = COLORS.medium;
    else if (threat === "LOW") color = COLORS.low;

    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;

    if (type === "radar") {
      // Diamond
      ctx.beginPath();
      ctx.moveTo(x, y - 5); ctx.lineTo(x + 4, y);
      ctx.lineTo(x, y + 5); ctx.lineTo(x - 4, y);
      ctx.closePath(); ctx.stroke();
    } else if (type === "thermal") {
      // Triangle
      ctx.beginPath();
      ctx.moveTo(x, y - 5); ctx.lineTo(x + 4, y + 4);
      ctx.lineTo(x - 4, y + 4); ctx.closePath(); ctx.stroke();
    } else if (type === "fused" || type === "enhanced_fused") {
      // Filled circle
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Camera: small square
      ctx.strokeRect(x - 3, y - 3, 6, 6);
    }

    // Draw label (intent / geo) for fused tracks
    if (label) {
      ctx.font = "8px Consolas, monospace";
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.8;
      ctx.fillText(label, x + 7, y + 3);
      ctx.globalAlpha = 1.0;
    }
  }

  // ---------------------------------------------------------------
  // Main update — called at full frame rate with interpolation
  // ---------------------------------------------------------------
  function update(data, prevData, t) {
    drawBackground();

    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;
    const radius = Math.min(cx, cy) - 10;

    const tracks = data.tracks || {};
    var prevTracks = (prevData && prevData.tracks) ? prevData.tracks : {};

    // Build prev lookups for interpolation
    var prevRadar = buildLookup(prevTracks.radar);
    var prevThermal = buildLookup(prevTracks.thermal);
    var prevFused = buildLookup(prevTracks.enhanced_fused);

    // Radar tracks — interpolated
    (tracks.radar || []).forEach(function (tr) {
      var key = tr.track_id;
      var prev = prevRadar[key];
      var range_m = tr.range_m;
      var az = tr.azimuth_deg;
      if (prev && prev.range_m != null && prev.azimuth_deg != null) {
        range_m = lerp(prev.range_m, tr.range_m, t);
        az = lerpAngle(prev.azimuth_deg, tr.azimuth_deg, t);
      }
      plotTrack(cx, cy, radius, range_m, az, "radar", tr.state, null, null);
    });

    // Thermal tracks — interpolated azimuth
    (tracks.thermal || []).forEach(function (tr) {
      var key = tr.track_id;
      var prev = prevThermal[key];
      var range_m = tr.range_m || (tr.position ? Math.sqrt(tr.position[0] ** 2 + tr.position[1] ** 2) : null);
      var az = tr.azimuth_deg;
      if (prev && prev.azimuth_deg != null && az != null) {
        az = lerpAngle(prev.azimuth_deg, tr.azimuth_deg, t);
      }
      plotTrack(cx, cy, radius, range_m, az, "thermal", tr.state, null, null);
    });

    // Enhanced fused tracks — interpolated, with intent + geo labels
    (tracks.enhanced_fused || []).forEach(function (tr) {
      var key = tr.fused_id;
      var prev = prevFused[key];
      var range_m = tr.range_m;
      var az = tr.azimuth_deg;
      if (prev && prev.range_m != null && prev.azimuth_deg != null) {
        range_m = lerp(prev.range_m, tr.range_m, t);
        az = lerpAngle(prev.azimuth_deg, tr.azimuth_deg, t);
      }
      // Build label: intent + geo coords
      var parts = [];
      if (tr.intent && tr.intent !== "unknown") parts.push(tr.intent.toUpperCase());
      if (tr.position_geo) {
        parts.push(tr.position_geo.lat.toFixed(3) + "N " + tr.position_geo.lon.toFixed(3) + "E");
      }
      var label = parts.length > 0 ? parts.join(" | ") : null;
      plotTrack(cx, cy, radius, range_m, az, "enhanced_fused", null, tr.threat_level, label);
    });

    // Plain fused tracks
    (tracks.fused || []).forEach(function (tr) {
      plotTrack(cx, cy, radius, tr.range_m, tr.azimuth_deg, "fused", null, null, null);
    });
  }

  return { update: update, resize: resize };
})();
