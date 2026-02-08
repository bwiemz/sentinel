/* SENTINEL HUD Video Feed */

window.HudFeed = (function () {
  const img = document.getElementById("hud-feed");
  const body = document.getElementById("hud-body");
  const toggle = document.getElementById("hud-toggle");
  let visible = true;

  function init() {
    img.src = "/api/video/hud";

    toggle.addEventListener("click", function () {
      visible = !visible;
      body.classList.toggle("hidden", !visible);
      toggle.textContent = visible ? "HIDE" : "SHOW";
    });
  }

  return { init: init };
})();
