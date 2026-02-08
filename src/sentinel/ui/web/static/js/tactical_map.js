/* SENTINEL Tactical Map — PPI Radar Scope */

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
      // Range label
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

  function plotTrack(cx, cy, radius, range_m, azimuth_deg, type, state, threat) {
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
  }

  function update(data) {
    resize();
    drawBackground();

    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;
    const radius = Math.min(cx, cy) - 10;

    const tracks = data.tracks || {};

    // Radar tracks
    (tracks.radar || []).forEach(function (t) {
      plotTrack(cx, cy, radius, t.range_m, t.azimuth_deg, "radar", t.state, null);
    });

    // Thermal tracks
    (tracks.thermal || []).forEach(function (t) {
      const range_m = t.range_m || (t.position ? Math.sqrt(t.position[0] ** 2 + t.position[1] ** 2) : null);
      plotTrack(cx, cy, radius, range_m, t.azimuth_deg, "thermal", t.state, null);
    });

    // Enhanced fused tracks
    (tracks.enhanced_fused || []).forEach(function (t) {
      plotTrack(cx, cy, radius, t.range_m, t.azimuth_deg, "enhanced_fused", null, t.threat_level);
    });

    // Fused tracks
    (tracks.fused || []).forEach(function (t) {
      plotTrack(cx, cy, radius, t.range_m, t.azimuth_deg, "fused", null, null);
    });
  }

  return { update: update, resize: resize };
})();
