/* SENTINEL Sensor Status Panel */

window.SensorPanel = (function () {
  const container = document.getElementById("sensor-body");

  const SENSOR_NAMES = {
    camera: "CAMERA",
    detector: "DETECTOR",
    radar: "RADAR",
    multifreq_radar: "MF RADAR",
    thermal: "THERMAL",
    quantum_radar: "QI RADAR",
    fusion: "FUSION",
  };

  function update(status) {
    const health = status.sensor_health || {};
    let html = "";

    for (const [key, label] of Object.entries(SENSOR_NAMES)) {
      const info = health[key];
      if (!info) continue;

      let dotClass = "offline";
      if (info.enabled && info.error_count === 0) dotClass = "online";
      else if (info.enabled && info.error_count > 0) dotClass = "degraded";

      html +=
        '<div class="sensor-card">' +
        '<div class="sensor-dot ' + dotClass + '"></div>' +
        '<div class="sensor-name">' + label + "</div>" +
        '<div class="sensor-errors">' +
        (info.error_count > 0 ? "ERR: " + info.error_count : "OK") +
        "</div></div>";
    }

    container.innerHTML = html;
  }

  return { update: update };
})();
