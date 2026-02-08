/* SENTINEL Track Table — Sortable */

window.TrackTable = (function () {
  const tbody = document.getElementById("tracks-tbody");
  const headers = document.querySelectorAll("#tracks-table th");
  let currentSort = "id";
  let sortAsc = true;
  let lastRows = [];

  // Column sort handlers
  headers.forEach(function (th) {
    th.addEventListener("click", function () {
      const field = th.dataset.sort;
      if (currentSort === field) {
        sortAsc = !sortAsc;
      } else {
        currentSort = field;
        sortAsc = true;
      }
      headers.forEach(function (h) { h.classList.remove("sorted-asc", "sorted-desc"); });
      th.classList.add(sortAsc ? "sorted-asc" : "sorted-desc");
      render(lastRows);
    });
  });

  function buildRows(data) {
    const tracks = data.tracks || {};
    const rows = [];

    (tracks.camera || []).forEach(function (t) {
      rows.push({
        id: (t.track_id || "").substring(0, 8),
        type: "CAM",
        state: t.state || "",
        range: null,
        azimuth: null,
        velocity: t.velocity ? Math.sqrt(t.velocity[0] ** 2 + t.velocity[1] ** 2).toFixed(1) : "",
        score: t.score != null ? t.score.toFixed(2) : "",
        threat: "",
        _state: t.state,
      });
    });

    (tracks.radar || []).forEach(function (t) {
      rows.push({
        id: (t.track_id || "").substring(0, 8),
        type: "RDR",
        state: t.state || "",
        range: t.range_m != null ? (t.range_m / 1000).toFixed(1) : "",
        azimuth: t.azimuth_deg != null ? t.azimuth_deg.toFixed(1) : "",
        velocity: t.velocity_mps ? Math.sqrt(t.velocity_mps[0] ** 2 + t.velocity_mps[1] ** 2).toFixed(0) : "",
        score: t.score != null ? t.score.toFixed(2) : "",
        threat: "",
        _state: t.state,
      });
    });

    (tracks.thermal || []).forEach(function (t) {
      rows.push({
        id: (t.track_id || "").substring(0, 8),
        type: "THM",
        state: t.state || "",
        range: null,
        azimuth: t.azimuth_deg != null ? t.azimuth_deg.toFixed(1) : "",
        velocity: "",
        score: t.score != null ? t.score.toFixed(2) : "",
        threat: "",
        _state: t.state,
      });
    });

    (tracks.enhanced_fused || []).forEach(function (t) {
      rows.push({
        id: (t.fused_id || "").substring(0, 8),
        type: "FUS",
        state: "",
        range: t.range_m != null ? (t.range_m / 1000).toFixed(1) : "",
        azimuth: t.azimuth_deg != null ? t.azimuth_deg.toFixed(1) : "",
        velocity: t.velocity_mps != null ? t.velocity_mps.toFixed(0) : "",
        score: t.fusion_quality != null ? t.fusion_quality.toFixed(2) : "",
        threat: t.threat_level || "",
        _state: "",
      });
    });

    return rows;
  }

  function sortRows(rows) {
    return rows.slice().sort(function (a, b) {
      let va = a[currentSort], vb = b[currentSort];
      if (va == null) va = "";
      if (vb == null) vb = "";
      const na = parseFloat(va), nb = parseFloat(vb);
      let cmp;
      if (!isNaN(na) && !isNaN(nb)) cmp = na - nb;
      else cmp = String(va).localeCompare(String(vb));
      return sortAsc ? cmp : -cmp;
    });
  }

  function threatColor(level) {
    if (level === "CRITICAL") return "var(--color-critical)";
    if (level === "HIGH") return "var(--color-high)";
    if (level === "MEDIUM") return "var(--color-medium)";
    if (level === "LOW") return "var(--color-low)";
    return "";
  }

  function render(rows) {
    const sorted = sortRows(rows);
    let html = "";
    for (const r of sorted) {
      const cls = r._state ? "track-row-" + r._state : "";
      html += '<tr class="' + cls + '">' +
        "<td>" + r.id + "</td>" +
        "<td>" + r.type + "</td>" +
        "<td>" + r.state + "</td>" +
        "<td>" + (r.range || "-") + "</td>" +
        "<td>" + (r.azimuth || "-") + "</td>" +
        "<td>" + (r.velocity || "-") + "</td>" +
        "<td>" + r.score + "</td>" +
        '<td style="color:' + threatColor(r.threat) + '">' + r.threat + "</td>" +
        "</tr>";
    }
    tbody.innerHTML = html;
  }

  function update(data) {
    lastRows = buildRows(data);
    render(lastRows);
  }

  return { update: update };
})();
