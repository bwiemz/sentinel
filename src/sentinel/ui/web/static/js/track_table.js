/* SENTINEL Track Table — Sortable, incremental DOM updates */

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
        intent: "",
        iff: "",
        roe: "",
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
        velocity: t.velocity_mps != null ? (typeof t.velocity_mps === "number" ? t.velocity_mps.toFixed(0) : Math.sqrt(t.velocity_mps[0] ** 2 + t.velocity_mps[1] ** 2).toFixed(0)) : "",
        score: t.score != null ? t.score.toFixed(2) : "",
        threat: "",
        intent: "",
        iff: "",
        roe: "",
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
        intent: "",
        iff: "",
        roe: "",
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
        intent: t.intent && t.intent !== "unknown" ? t.intent.toUpperCase() : "",
        iff: t.iff_identification && t.iff_identification !== "unknown" ? t.iff_identification.toUpperCase().replace(/_/g, " ") : "",
        roe: t.engagement_auth && t.engagement_auth !== "weapons_hold" ? t.engagement_auth.toUpperCase().replace(/_/g, " ") : "",
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

  function intentColor(intent) {
    if (intent === "ATTACK") return "var(--color-critical)";
    if (intent === "APPROACH") return "var(--color-high)";
    if (intent === "EVASION") return "var(--color-medium)";
    if (intent === "PATROL") return "var(--color-radar)";
    if (intent === "TRANSIT") return "var(--color-text-dim)";
    return "";
  }

  function iffColor(iff) {
    if (iff === "FRIENDLY") return "var(--color-iff-friendly)";
    if (iff === "ASSUMED FRIENDLY") return "var(--color-iff-assumed-friendly)";
    if (iff === "HOSTILE") return "var(--color-iff-hostile)";
    if (iff === "ASSUMED HOSTILE") return "var(--color-iff-assumed-hostile)";
    if (iff === "SPOOF SUSPECT") return "var(--color-iff-spoof)";
    if (iff === "PENDING") return "var(--color-iff-pending)";
    return "";
  }

  function roeColor(roe) {
    if (roe === "WEAPONS FREE") return "var(--color-auth-weapons-free)";
    if (roe === "WEAPONS TIGHT") return "var(--color-auth-weapons-tight)";
    if (roe === "HOLD FIRE") return "var(--color-auth-hold-fire)";
    return "";
  }

  var COL_COUNT = 11;
  var FIELDS = ["id", "type", "state", "range", "azimuth", "velocity", "score", "threat", "intent", "iff", "roe"];

  function ensureRow(index) {
    if (index < tbody.children.length) return tbody.children[index];
    var tr = document.createElement("tr");
    for (var j = 0; j < COL_COUNT; j++) {
      tr.appendChild(document.createElement("td"));
    }
    tbody.appendChild(tr);
    return tr;
  }

  function render(rows) {
    var sorted = sortRows(rows);

    for (var i = 0; i < sorted.length; i++) {
      var r = sorted[i];
      var tr = ensureRow(i);

      var cls = r._state ? "track-row-" + r._state : "";
      if (tr.className !== cls) tr.className = cls;

      var cells = tr.children;
      for (var j = 0; j < FIELDS.length; j++) {
        var val = r[FIELDS[j]];
        var text = (val != null && val !== "") ? String(val) : "-";
        if (FIELDS[j] === "threat") {
          // threat column — also set color
          text = val || "";
          if (cells[j].textContent !== text) cells[j].textContent = text;
          var tc = threatColor(val);
          if (cells[j].style.color !== tc) cells[j].style.color = tc;
        } else if (FIELDS[j] === "intent") {
          // intent column — color by intent type
          text = val || "";
          if (cells[j].textContent !== text) cells[j].textContent = text;
          var ic = intentColor(val);
          if (cells[j].style.color !== ic) cells[j].style.color = ic;
        } else if (FIELDS[j] === "iff") {
          text = val || "";
          if (cells[j].textContent !== text) cells[j].textContent = text;
          var ifc = iffColor(val);
          if (cells[j].style.color !== ifc) cells[j].style.color = ifc;
        } else if (FIELDS[j] === "roe") {
          text = val || "";
          if (cells[j].textContent !== text) cells[j].textContent = text;
          var rc = roeColor(val);
          if (cells[j].style.color !== rc) cells[j].style.color = rc;
        } else {
          if (cells[j].textContent !== text) cells[j].textContent = text;
        }
      }
    }

    // Remove excess rows
    while (tbody.children.length > sorted.length) {
      tbody.removeChild(tbody.lastChild);
    }
  }

  function update(data) {
    lastRows = buildRows(data);
    render(lastRows);
  }

  return { update: update };
})();
