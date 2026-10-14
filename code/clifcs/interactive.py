#!/usr/bin/env python3

"""
an interactive PaCMAP for the report: hover a patient, see their guideline fit

Rendered as a self-contained HTML fragment -- a canvas, a compact columnar JSON
payload and about a hundred lines of vanilla JavaScript. No plotting library,
for the same reason the rest of the report inlines its figures as base64: the
report has to open from anywhere, years from now, with nothing fetched at view
time. A plotly bundle would add three megabytes of JavaScript and a CDN
dependency to a document that currently has neither.

Canvas rather than SVG because 34,142 circles is far past where SVG stops being
usable in a browser. Hit-testing is done against a uniform grid built once at
load, so hover stays constant-time rather than scanning every point.

ON COLOUR. Nine guideline assignments cannot be separated by colour -- the
palette validates three slots under the all-pairs gate a scatter needs. That
gate is about colour being the *sole* carrier of identity, and here it is not:
every point names its guideline on hover, the legend is clickable to isolate one
guideline at a time, and the counts are printed beside it. Colour is the cue and
the interaction is the channel. The static small-multiple figures remain the
answer for anyone reading on paper.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from clifcs.palette import CATEGORICAL_8, CONTEXT, GRID, INK_PRIMARY, INK_SECONDARY

#: colour per guideline, in the trials' chronological order, with grey last for
#: the patients no guideline fits
UNASSIGNED = "#8a8980"


def _rounded(values, digits: int = 2) -> list:
    out = []
    for value in values:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            out.append(None)
        else:
            out.append(round(float(value), digits))
    return out


def build_payload(frame: pl.DataFrame, trials, projection: str = "pacmap") -> dict:
    """
    the columnar JSON the page reads

    Columnar, not row-wise: 34,142 objects with fifteen named keys each is about
    four times the bytes of fifteen arrays, and the report is already large.
    Coordinates are rounded to two decimals and rates to three, which costs
    nothing visible and roughly halves the payload again.
    """
    x, y = f"{projection}_1", f"{projection}_2"
    keys = [t.key for t in trials]
    labels = [t.label for t in trials]

    assigned = frame["assigned_guideline"].to_list()
    index = {label: i for i, label in enumerate(labels)}
    group = [index.get(a, len(labels)) if a is not None else len(labels) for a in assigned]

    def column(name, digits=2):
        if name not in frame.columns:
            return None
        return _rounded(frame[name].to_list(), digits)

    def flags(prefix):
        """per patient, a bitmask over trials -- far smaller than 8 arrays"""
        masks = []
        columns = [
            frame[f"{prefix}{key}"].fill_null(False).to_list()
            if f"{prefix}{key}" in frame.columns
            else [False] * frame.height
            for key in keys
        ]
        for row in range(frame.height):
            bits = 0
            for position in range(len(keys)):
                if columns[position][row]:
                    bits |= 1 << position
            masks.append(bits)
        return masks

    payload = {
        "labels": labels + ["no guideline fits"],
        "colors": list(CATEGORICAL_8[: len(labels)]) + [UNASSIGNED],
        "x": column(x),
        "y": column(y),
        "g": group,
        "site": frame["site"].to_list(),
        "age": [int(a) if a is not None else None for a in frame["age_at_admission"]],
        "sex": [s[0] if s else "?" for s in frame["sex_category"].fill_null("?")],
        "died": [bool(v) for v in frame["in_hospital_mortality"].fill_null(False)],
        "los": column("icu_los_days", 1),
        "sofa": [
            int(v) if v is not None else None for v in frame["sofa_total"].fill_null(-1)
        ],
        "sofa_ncv": [
            int(v) if v is not None else None
            for v in frame["sofa_non_cv"].fill_null(-1)
        ],
        "sbp": column("sbp_at_onset", 0),
        "map": column("map_at_onset", 0),
        "lact": column("lactate", 1),
        "shock": [str(s) for s in frame["shock_type"].fill_null("none coded")],
        "elig": flags("elig_"),
        "inside": flags("inside_t1_"),
        "conc": flags("concordant_"),
        "nfit": [int(v) for v in frame["n_specific_trials"].fill_null(0)],
    }
    return payload


def render(
    frame: pl.DataFrame, trials, projection: str = "pacmap", uid: str = "a"
) -> str:
    """
    the HTML fragment: canvas, legend, tooltip and the script that drives it

    `uid` must differ between fragments embedded in the same document. The
    report shows one per anchor, and with a shared element id the second
    fragment's script binds to the first fragment's canvas -- the onset tab
    rendered but did nothing.
    """
    payload = build_payload(frame, trials, projection)
    data = json.dumps(payload, separators=(",", ":"))
    pretty = projection.upper()

    counts = {}
    for g in payload["g"]:
        counts[g] = counts.get(g, 0) + 1
    legend = "".join(
        f'<button class="gl" data-g="{i}" style="--c:{c}">'
        f'<span class="sw"></span>{label}'
        f'<span class="n">{counts.get(i, 0):,}</span></button>'
        for i, (label, c) in enumerate(zip(payload["labels"], payload["colors"]))
    )

    return f"""
<div class="ipx">
  <div class="ipx-head">
    <div class="ipx-legend" id="ipx-legend-{uid}">{legend}
      <button class="gl gl-all" data-g="-1"><span class="sw sw-all"></span>all</button>
    </div>
    <p class="ipx-hint">Hover a patient for their detail. Click a guideline to
    isolate it; click <i>all</i> to restore. Each point is one patient at the
    anchor, positioned by {pretty} of the 1024-d embedding.</p>
  </div>
  <div class="ipx-stage">
    <canvas id="ipx-canvas-{uid}" width="1280" height="820"></canvas>
    <div id="ipx-tip-{uid}" class="ipx-tip"></div>
  </div>
</div>
<style>
/* repeated per fragment and identical; harmless, and keeping it local
   means a fragment pasted anywhere still styles itself */
.ipx {{ margin: 1.2rem 0 1.6rem; }}
.ipx-legend {{ display:flex; flex-wrap:wrap; gap:.3rem .5rem; margin-bottom:.4rem; }}
.ipx .gl {{ display:inline-flex; align-items:center; gap:.35rem; border:1px solid {GRID};
  background:transparent; border-radius:3px; padding:.16rem .45rem; cursor:pointer;
  font:inherit; font-size:.72rem; color:{INK_SECONDARY}; }}
.ipx .gl[aria-pressed="true"] {{ border-color:{INK_PRIMARY}; color:{INK_PRIMARY}; }}
.ipx .sw {{ width:.6rem; height:.6rem; border-radius:50%; background:var(--c); }}
.ipx .sw-all {{ background:linear-gradient(90deg,{CATEGORICAL_8[0]},{CATEGORICAL_8[2]}); }}
.ipx .n {{ color:{INK_SECONDARY}; font-variant-numeric:tabular-nums; opacity:.75; }}
.ipx-hint {{ font-size:.72rem; color:{INK_SECONDARY}; margin:.2rem 0 .5rem; }}
.ipx-stage {{ position:relative; }}
.ipx canvas {{ width:100%; height:auto; display:block; cursor:crosshair; }}
.ipx-tip {{ position:absolute; pointer-events:none; display:none; z-index:5;
  background:#fff; border:1px solid {GRID}; border-radius:4px; padding:.45rem .6rem;
  font-size:.72rem; line-height:1.45; color:{INK_PRIMARY}; max-width:19rem;
  box-shadow:0 2px 10px rgba(0,0,0,.09); }}
.ipx-tip b {{ font-weight:600; }}
.ipx-tip .r {{ color:{INK_SECONDARY}; }}
</style>
<script>
(function() {{
  const D = {data};
  const cv = document.getElementById('ipx-canvas-{uid}');
  const tip = document.getElementById('ipx-tip-{uid}');
  const ctx = cv.getContext('2d');
  const N = D.x.length, W = cv.width, H = cv.height, PAD = 26;
  let only = -1;

  const xs = D.x, ys = D.y;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < N; i++) {{
    if (xs[i] === null) continue;
    if (xs[i] < x0) x0 = xs[i]; if (xs[i] > x1) x1 = xs[i];
    if (ys[i] < y0) y0 = ys[i]; if (ys[i] > y1) y1 = ys[i];
  }}
  // one unit of data is one unit of screen on both axes, so the projection is
  // not stretched -- the same 1:1 rule the static figures keep
  const s = Math.min((W - 2*PAD) / (x1 - x0), (H - 2*PAD) / (y1 - y0));
  const ox = (W - s*(x1 - x0))/2, oy = (H - s*(y1 - y0))/2;
  const px = i => ox + (xs[i] - x0)*s;
  const py = i => H - (oy + (ys[i] - y0)*s);

  // uniform grid for hit-testing, built once
  const CELL = 14, gw = Math.ceil(W/CELL), gh = Math.ceil(H/CELL);
  const grid = new Array(gw*gh);
  for (let i = 0; i < N; i++) {{
    if (xs[i] === null) continue;
    const c = Math.floor(px(i)/CELL), r = Math.floor(py(i)/CELL);
    const k = r*gw + c;
    if (!grid[k]) grid[k] = [];
    grid[k].push(i);
  }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    // unselected first, so the isolated guideline is never buried
    for (let pass = 0; pass < 2; pass++) {{
      for (let i = 0; i < N; i++) {{
        if (xs[i] === null) continue;
        const sel = (only < 0) || (D.g[i] === only);
        if (pass === 0 ? sel : !sel) continue;
        ctx.fillStyle = sel ? D.colors[D.g[i]] : '{CONTEXT}';
        ctx.globalAlpha = sel ? 0.78 : 0.5;
        ctx.beginPath();
        ctx.arc(px(i), py(i), sel ? 1.9 : 1.2, 0, 6.2832);
        ctx.fill();
      }}
    }}
    ctx.globalAlpha = 1;
  }}

  function names(mask) {{
    const out = [];
    for (let i = 0; i < D.labels.length - 1; i++)
      if (mask & (1 << i)) out.push(D.labels[i]);
    return out;
  }}

  function show(i, mx, my) {{
    const el = [], ins = names(D.inside[i]), con = names(D.conc[i]);
    const eligible = names(D.elig[i]);
    const sofa = D.sofa[i] < 0 ? '?' : D.sofa[i];
    const sofaN = D.sofa_ncv[i] < 0 ? '?' : D.sofa_ncv[i];
    tip.innerHTML =
      '<b>' + D.labels[D.g[i]] + '</b><br>' +
      '<span class="r">' + D.site[i] + ' · ' + D.age[i] + D.sex[i] +
      ' · coded ' + D.shock[i] + '</span><br>' +
      '<span class="r">SBP</span> ' + (D.sbp[i] ?? '?') +
      ' <span class="r">MAP</span> ' + (D.map[i] ?? '?') +
      ' <span class="r">lactate</span> ' + (D.lact[i] ?? '?') + '<br>' +
      '<span class="r">SOFA</span> ' + sofa +
      ' <span class="r">(non-CV ' + sofaN + ')</span><br>' +
      '<span class="r">eligible for</span> ' +
      (eligible.length ? eligible.length + ': ' + eligible.join(', ') : 'none') + '<br>' +
      '<span class="r">inside published IQRs of</span> ' +
      (ins.length ? ins.join(', ') : 'none') + '<br>' +
      '<span class="r">treated per</span> ' +
      (con.length ? con.join(', ') : 'none') + '<br>' +
      '<b>' + (D.died[i] ? 'died in hospital' : 'survived') + '</b>' +
      ' <span class="r">· ICU ' + (D.los[i] ?? '?') + ' d</span>';
    tip.style.display = 'block';
    const r = cv.getBoundingClientRect();
    const sx = r.width / W;
    tip.style.left = Math.min(mx + 14, r.width - tip.offsetWidth - 6) + 'px';
    tip.style.top = Math.max(my - tip.offsetHeight - 10, 4) + 'px';
  }}

  cv.addEventListener('mousemove', function(e) {{
    const r = cv.getBoundingClientRect();
    const mx = (e.clientX - r.left), my = (e.clientY - r.top);
    const cx = mx * (W / r.width), cy = my * (H / r.height);
    let best = -1, bd = 64;
    const c0 = Math.floor(cx/CELL), r0 = Math.floor(cy/CELL);
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {{
      const cell = grid[(r0+dr)*gw + (c0+dc)];
      if (!cell) continue;
      for (const i of cell) {{
        if (only >= 0 && D.g[i] !== only) continue;
        const d = (px(i)-cx)**2 + (py(i)-cy)**2;
        if (d < bd) {{ bd = d; best = i; }}
      }}
    }}
    if (best >= 0) show(best, mx, my); else tip.style.display = 'none';
  }});
  cv.addEventListener('mouseleave', () => tip.style.display = 'none');

  const legendEl = document.getElementById('ipx-legend-{uid}');
  legendEl.querySelectorAll('.gl').forEach(function(b) {{
    b.addEventListener('click', function() {{
      const g = parseInt(b.dataset.g, 10);
      only = (g === only) ? -1 : g;
      legendEl.querySelectorAll('.gl').forEach(o =>
        o.setAttribute('aria-pressed', String(parseInt(o.dataset.g, 10) === only)));
      draw();
    }});
  }});
  draw();
}})();
</script>
"""
