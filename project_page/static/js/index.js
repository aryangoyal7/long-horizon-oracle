// Adapted from action-chunking.github.io/static/js/index.js: the section rail
// (initTableOfContents), KaTeX auto-render (initAbstractMath) and the rollout
// grid with numbered buttons (initDroidResultsGrid -> initLabelVideoGrid).

function onReady(fn) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", fn);
  } else {
    fn();
  }
}

onReady(function() {
  initAbstractMath();
  initLabelVideoGrid();
  initTableOfContents();
});

window.addEventListener("load", initAbstractMath);
window.setTimeout(initAbstractMath, 1200);

// Left-rail table of contents: highlight the section whose top has scrolled
// past a trigger line near the top of the viewport, and hide the whole rail
// until the first listed section is reached.
function initTableOfContents() {
  var nav = document.getElementById("toc-nav");
  if (!nav) { return; }
  var secs = Array.prototype.slice.call(nav.querySelectorAll(".toc-item")).map(function(item) {
    var anchor = document.getElementById(item.getAttribute("href").slice(1));
    return { item: item, section: anchor ? anchor.closest("section") : null };
  }).filter(function(s) { return s.section; });

  function update() {
    var trigger = window.scrollY + window.innerHeight * 0.35;
    var current = null;
    secs.forEach(function(s) { if (s.section.offsetTop <= trigger) { current = s; } });
    secs.forEach(function(s) { s.item.classList.toggle("active", s === current); });
    nav.style.opacity = current ? "1" : "0";
    nav.style.pointerEvents = current ? "auto" : "none";
  }

  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

var mathRenderAttempts = 0;
function initAbstractMath() {
  if (typeof renderMathInElement !== "function") {
    if (mathRenderAttempts < 12) {
      mathRenderAttempts += 1;
      window.setTimeout(initAbstractMath, 300);
    }
    return;
  }

  document.querySelectorAll(".math-content, #abstract-content").forEach(function(el) {
    if (el.getAttribute("data-math-rendered") === "true") {
      return;
    }
    renderMathInElement(el, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
      throwOnError: false
    });
    el.setAttribute("data-math-rendered", "true");
  });
}

// One card per (platform, task); six numbered buttons under the video swap
// its src, exactly as the reference's DROID grid does. Clip list and tooltips
// come from static/js/label_videos.js (written by tools/build_video_manifest.py).
function initLabelVideoGrid() {
  var track = document.getElementById("label-video-grid");

  if (!track || track.getAttribute("data-initialized") === "true" ||
      typeof LABEL_VIDEO_TASKS === "undefined") {
    return;
  }
  track.setAttribute("data-initialized", "true");

  function videoPath(file) {
    return "./static/videos/" + file;
  }

  function playVideo(video) {
    var playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(function() {});
    }
  }

  LABEL_VIDEO_TASKS.forEach(function(task) {
    var card = document.createElement("div");
    card.className = "label-video-card";

    var title = document.createElement("h4");
    var plat = document.createElement("span");
    plat.className = "platform";
    plat.textContent = task.platformLabel + " · ";
    title.appendChild(plat);
    title.appendChild(document.createTextNode(task.task));
    card.appendChild(title);

    var first = task.clips.filter(function(c) { return c.file; })[0];
    if (!first) { return; }
    var video = document.createElement("video");
    video.controls = true;
    video.autoplay = true;
    video.muted = true;
    video.defaultMuted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.setAttribute("muted", "");
    video.setAttribute("autoplay", "");
    video.setAttribute("playsinline", "");
    video.src = videoPath(first.file);
    video.addEventListener("loadeddata", function() { playVideo(video); });
    card.appendChild(video);

    var selector = document.createElement("div");
    selector.className = "rollout-selector";
    selector.setAttribute("aria-label", task.task + " label selector");

    task.clips.forEach(function(clip, buttonIndex) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rollout-button cls-" + clip.cls + (clip === first ? " is-active" : "");
      btn.textContent = String(buttonIndex + 1);
      btn.title = clip.tip;
      btn.setAttribute("aria-label", task.task + " clip " + (buttonIndex + 1) + ": " + clip.tip);

      if (!clip.file) {
        btn.disabled = true;
      } else {
        (function(selectedClip, button) {
          button.addEventListener("click", function() {
            video.pause();
            video.src = videoPath(selectedClip.file);
            video.load();
            playVideo(video);
            selector.querySelectorAll(".rollout-button").forEach(function(otherButton) {
              otherButton.classList.toggle("is-active", otherButton === button);
            });
          });
        })(clip, btn);
      }
      selector.appendChild(btn);
    });

    card.appendChild(selector);
    track.appendChild(card);
    playVideo(video);
  });
}
