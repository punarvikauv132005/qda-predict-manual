/* ============================================================================
   Aviation Accident Prediction System - QDA (Manual Weather Entry)
   Front-end logic: form validation, animations, high-risk alert
   ========================================================================== */

/* ------------------------------------------------------------
   Airport database (display only - for reference)
   ------------------------------------------------------------ */
const AIRPORTS = [
  { code: "KJFK", name: "New York JFK (USA)" },
  { code: "KLAX", name: "Los Angeles LAX (USA)" },
  { code: "KORD", name: "Chicago O'Hare (USA)" },
  { code: "KATL", name: "Atlanta Hartsfield (USA)" },
  { code: "KDFW", name: "Dallas Fort Worth (USA)" },
  { code: "KDEN", name: "Denver International (USA)" },
  { code: "KMIA", name: "Miami International (USA)" },
  { code: "KSFO", name: "San Francisco (USA)" },
  { code: "EGLL", name: "London Heathrow (UK)" },
  { code: "LFPG", name: "Paris Charles de Gaulle (FR)" },
  { code: "EDDF", name: "Frankfurt Airport (DE)" },
  { code: "EHAM", name: "Amsterdam Schiphol (NL)" },
  { code: "LEMD", name: "Madrid Barajas (ES)" },
  { code: "RJTT", name: "Tokyo Haneda (JP)" },
  { code: "ZBAA", name: "Beijing Capital (CN)" },
  { code: "VABB", name: "Mumbai Chhatrapati (IN)" },
  { code: "VIDP", name: "New Delhi Indira Gandhi (IN)" },
  { code: "VOBL", name: "Bengaluru Kempegowda (IN)" },
  { code: "VOMM", name: "Chennai International (IN)" },
  { code: "YSSY", name: "Sydney Kingsford Smith (AU)" },
  { code: "OMDB", name: "Dubai International (AE)" },
  { code: "SBGR", name: "São Paulo Guarulhos (BR)" },
];

/* ------------------------------------------------------------
   Utility helpers
   ------------------------------------------------------------ */
function $(id) { return document.getElementById(id); }

/* ------------------------------------------------------------
   Populate the airport dropdown (reference only)
   ------------------------------------------------------------ */
function populateAirports() {
  const select = $("airport");
  if (!select) return;
  AIRPORTS.forEach((ap) => {
    const option = document.createElement("option");
    option.value = ap.code;
    option.textContent = `${ap.name} (${ap.code})`;
    select.appendChild(option);
  });
}

/* ------------------------------------------------------------
   Form validation (client side)
   ------------------------------------------------------------ */
function validateForm(form) {
  const inputs = form.querySelectorAll("input[type=number]");
  const errors = [];

  inputs.forEach((input) => {
    if (!input.value || input.value === "") {
      errors.push(`${input.name} is required.`);
      input.classList.add("is-invalid");
    } else {
      input.classList.remove("is-invalid");
    }
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const val = parseFloat(input.value);
    if (!isNaN(val) && (val < min || val > max)) {
      errors.push(`${input.name} must be between ${min} and ${max}.`);
      input.classList.add("is-invalid");
    }
  });

  const aircraftType = $("aircraft_type");
  if (aircraftType && !aircraftType.value) {
    errors.push("Please select an aircraft type.");
    aircraftType.classList.add("is-invalid");
  }

  return errors;
}

/* ------------------------------------------------------------
   Prediction form submit: validate, show spinner, then POST
   ------------------------------------------------------------ */
function initPredictionForm() {
  const form = $("predictionForm");
  const overlay = $("loadingOverlay");
  const btn = $("predictBtn");
  if (!form) return;

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    const errors = validateForm(form);
    if (errors.length > 0) {
      alert("Please fix the following issues:\n\n- " + errors.join("\n- "));
      return;
    }

    // Show the loading overlay, then submit the form normally
    if (btn) btn.disabled = true;
    if (overlay) overlay.classList.add("active");
    setTimeout(() => {
      form.submit();
    }, 600);
  });
}

/* ------------------------------------------------------------
   High-risk alert sound (Web Audio API)
   ------------------------------------------------------------ */
function playHighRiskAlert() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const play = () => {
      [0, 0.3, 0.6].forEach((start, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = "square";
        osc.frequency.value = 880 + (i % 2) * 220;
        gain.gain.setValueAtTime(0.001, ctx.currentTime + start);
        gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + start + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + 0.2);
        osc.start(ctx.currentTime + start);
        osc.stop(ctx.currentTime + start + 0.25);
      });
    };
    if (ctx.state === "suspended") {
      ctx.resume().then(play);
    } else {
      play();
    }
  } catch (e) {
    console.warn("Audio alert not supported:", e);
  }
}

/* ------------------------------------------------------------
   Reveal-on-scroll
   ------------------------------------------------------------ */
function initReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
}

/* ------------------------------------------------------------
   Animated counters (stats strip)
   ------------------------------------------------------------ */
function initCounters() {
  const counters = document.querySelectorAll(".stat-value[data-count]");
  if (!counters.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = parseInt(el.dataset.count, 10);
        const duration = 1400;
        const start = performance.now();
        const step = (now) => {
          const progress = Math.min((now - start) / duration, 1);
          el.textContent = Math.round(progress * target).toLocaleString();
          if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
        observer.unobserve(el);
      });
    },
    { threshold: 0.4 }
  );
  counters.forEach((el) => observer.observe(el));
}

/* ------------------------------------------------------------
   Navbar shadow on scroll
   ------------------------------------------------------------ */
function initNavbarScroll() {
  const nav = document.querySelector(".aviation-navbar");
  if (!nav) return;
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 40);
  window.addEventListener("scroll", onScroll);
  onScroll();
}

/* ------------------------------------------------------------
   Initialise everything on page load
   ------------------------------------------------------------ */
document.addEventListener("DOMContentLoaded", function () {
  populateAirports();
  initPredictionForm();
  initReveal();
  initCounters();
  initNavbarScroll();
});
