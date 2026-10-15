/* S1-Core UI — render-only frontend.
 *
 * Law: this file never computes anything that matters. Every verdict,
 * sweep, fit, drift figure, and chain check comes from the server, which
 * computes it with s1core. What is drawn here is what replay defends:
 * even the pass/fail ticks on the gauges are re-presentations of the
 * server's verdict against the record's own gate inputs.
 */

const $app = document.getElementById('app');
const $drawerRoot = document.getElementById('drawer-root');
const $badge = document.getElementById('backend-badge');
const $footerMeta = document.getElementById('footer-meta');

const STATUS_CLASS = {
  RATIFIED: 'RATIFIED',
  ABSTAINED_MODEL: 'ABSTAINED_MODEL',
  ABSTAINED_GATE: 'ABSTAINED_GATE',
  BIMODAL_CONFLICT: 'BIMODAL_CONFLICT',
  SKIPPED_DEPENDENCY: 'SKIPPED_DEPENDENCY',
  LOOP_GUARD_TRIPPED: 'LOOP_GUARD_TRIPPED',
};

const STATUS_HELP = {
  RATIFIED: 'a policy passed; the value was released',
  ABSTAINED_MODEL: 'the model chose an escape hatch',
  ABSTAINED_GATE: 'rejected by margin / entropy / conformal / divergence',
  BIMODAL_CONFLICT: 'score distribution is polarized; no average was collapsed',
  SKIPPED_DEPENDENCY: 'a wave-DAG parent did not ratify with the required value',
  LOOP_GUARD_TRIPPED: 'too many consecutive abstentions on this stream',
};

const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmt = (x, d = 3) => (x === null || x === undefined) ? '—' : Number(x).toFixed(d);
/* display-only float cleanup: 3.8700000000000006 -> 3.87 (the ledger keeps full precision) */
const fmtValue = (v) => typeof v === 'number' ? JSON.stringify(Number(v.toFixed(4))) : JSON.stringify(v);

async function api(path, body) {
  const res = await fetch(path, body !== undefined ? {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function chip(status, extra = '') {
  const cls = STATUS_CLASS[status] || 'SKIPPED_DEPENDENCY';
  return `<span class="chip ${cls}${extra}" title="${esc(STATUS_HELP[status] || '')}">${esc(status)}</span>`;
}

/* ------------------------------------------------------------------ state */

const S = {
  state: null,
  scenarios: [],
  runs: new Map(),       // run_id -> payload (cache)
  recent: [],            // [{run_id, kind, label}]
  selected: null,        // {runId, idx}
  composer: null,        // persistent composer draft
};

async function cacheRun(id) {
  if (!S.runs.has(id)) S.runs.set(id, await api(`/api/runs/${id}`));
  return S.runs.get(id);
}

function rememberRun(payload, label) {
  S.runs.set(payload.run_id, payload);
  S.recent.unshift({run_id: payload.run_id, kind: payload.kind, label});
  S.recent = S.recent.slice(0, 12);
}

/* ------------------------------------------------------------------ router */

function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const parts = h.split('/').filter(Boolean);
  if (parts[0] === 'run') return {view: 'run', id: parts[1], idx: parts[2] ?? null};
  if (parts[0] === 'composer') return {view: 'composer'};
  return {view: 'home'};
}

function nav(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

async function route() {
  const r = parseHash();
  setActiveNav(r.view);
  try {
    if (r.view === 'home') renderHome();
    else if (r.view === 'composer') renderComposer();
    else if (r.view === 'run') await renderRun(r.id, r.idx === null ? null : Number(r.idx));
  } catch (err) {
    $app.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  }
}

function setActiveNav(view) {
  document.querySelectorAll('.topbar-nav a').forEach((a) => {
    a.classList.toggle('active', a.dataset.nav === view);
  });
}

/* ------------------------------------------------------------------ home */

function renderHome() {
  closeDrawer();
  const cards = S.scenarios.map((sc) => `
    <div class="card scenario-card" data-scenario="${esc(sc.id)}">
      <h3>${esc(sc.name)}</h3>
      <p>${esc(sc.description)}</p>
      <div class="meta">
        <span class="badge-det">${esc(sc.kind)}</span>
        ${sc.deterministic ? '<span class="badge-det">deterministic</span>' : '<span class="badge-det">live-capable</span>'}
        <button class="btn small primary" data-scenario-run="${esc(sc.id)}">Run →</button>
      </div>
    </div>`).join('');

  $app.innerHTML = `
    <div class="hero">
      <h1>Make the harness legible.</h1>
      <p>Verdicts with their gate boundaries drawn on the distribution, the
      wave pipeline as geometry, the ledger as a navigable spine, the
      admissibility gate as a live dial. Everything you see was computed by
      <span class="mono">s1core</span> — the browser only draws.</p>
    </div>
    <div class="section"><h2>Scenarios</h2><div class="grid cols-2">${cards}</div></div>
    <div class="section"><h2>Open a ledger <span class="hint">hash-chained JSONL under the repo (or --allow dirs)</span></h2>
      <div class="card">
        <div class="kv">
          <select id="ledger-select" aria-label="available ledgers"><option>scanning…</option></select>
          <button class="btn primary" id="ledger-open" disabled>Load &amp; verify</button>
        </div>
        <div class="kv" style="margin-top:8px">
          <input type="text" id="ledger-path" placeholder="…or type an exact path (e.g. examples/refund_triage_audit.jsonl)" spellcheck="false">
          <button class="btn" id="ledger-open-manual">Load path</button>
        </div>
        <div id="ledger-error"></div>
      </div>
    </div>
    ${S.recent.length ? `
      <div class="section"><h2>Recent runs</h2>
        <div class="card recent-list"><ul>${S.recent.map((r) =>
          `<li><a href="#/run/${r.run_id}">#${r.run_id}</a> <span class="dim">${esc(r.kind)}</span> — ${esc(r.label)}</li>`).join('')}</ul></div>
      </div>` : ''}
  `;

  $app.querySelectorAll('[data-scenario-run]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.stopPropagation();
      runScenario(el.dataset.scenarioRun);
    });
  });
  $app.querySelectorAll('.scenario-card').forEach((el) => {
    el.addEventListener('click', () => runScenario(el.dataset.scenario));
  });
  document.getElementById('ledger-open').addEventListener('click', () => {
    const selected = document.getElementById('ledger-select').value;
    if (selected) openLedger(selected);
  });
  document.getElementById('ledger-open-manual').addEventListener('click', () => {
    openLedger(document.getElementById('ledger-path').value.trim());
  });
  document.getElementById('ledger-path').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') openLedger(e.target.value.trim());
  });
  refreshLedgerList();
}

async function refreshLedgerList() {
  const select = document.getElementById('ledger-select');
  const button = document.getElementById('ledger-open');
  try {
    const { ledgers } = await api('/api/ledgers');
    if (!ledgers.length) {
      select.innerHTML = '<option value="">no audit ledgers found yet — run an example that writes one</option>';
      button.disabled = true;
      return;
    }
    select.innerHTML = ledgers.map((entry) =>
      `<option value="${esc(entry.path)}">${esc(entry.path)} · ${entry.records} record${entry.records === 1 ? '' : 's'}</option>`).join('');
    button.disabled = false;
  } catch (err) {
    select.innerHTML = `<option value="">could not list ledgers: ${esc(err.message)}</option>`;
    button.disabled = true;
  }
}

async function runScenario(id) {
  const meta = S.scenarios.find((s) => s.id === id);
  const card = $app.querySelector(`[data-scenario="${id}"]`);
  try {
    if (card) card.querySelector('button').textContent = 'running…';
    const backend = id === 'triage' ? 'auto' : 'toy';
    const payload = await api(`/api/scenarios/${id}/run`, {backend});
    rememberRun(payload, meta ? meta.name : id);
    nav(`#/run/${payload.run_id}`);
  } catch (err) {
    $app.insertAdjacentHTML('afterbegin',
      `<div class="error-box">${esc(err.message)}</div>`);
  } finally {
    if (card) card.querySelector('button').textContent = 'Run →';
  }
}

async function openLedger(path) {
  const box = document.getElementById('ledger-error');
  box.innerHTML = '';
  if (!path) {
    box.innerHTML = `<div class="error-box">no ledger selected</div>`;
    return;
  }
  try {
    const payload = await api('/api/ledger/load', {path});
    rememberRun(payload, `ledger ${payload.path.split('/').pop()}`);
    nav(`#/run/${payload.run_id}`);
  } catch (err) {
    box.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  }
}

/* ------------------------------------------------------------------ run view */

async function renderRun(id, inspectorIdx) {
  const run = await cacheRun(id);
  closeDrawer();
  S._lab = null;

  const sections = [];
  sections.push(verdictSection(run, inspectorIdx));
  if (run.waves) sections.push(pipelineSection(run));
  if (run.kind === 'certification') sections.push(certificationSection(run));
  if (run.kind === 'cascade') sections.push(cascadeSection(run));
  if (run.kind === 'recalibration') sections.push(recalibrationSection(run));
  if (Object.keys(run.labels || {}).length >= 4) sections.push(calibrationSection(run));
  if (run.records.length && run.records.some((r) => r.raw_probabilities
        && Object.keys(r.raw_probabilities).length)) {
    sections.push(gateLabSection(run));
  }
  sections.push(ledgerSection(run));

  $app.innerHTML = `
    <div class="section" style="margin-top:0">
      <h2 style="text-transform:none; letter-spacing:0; font-size:16px; color:var(--text)">
        ${runBadge(run)} <span class="dim" style="font-size:12.5px">
          run #${esc(run.run_id)} · ${esc(run.backend.kind)} · ${esc(run.backend.model_id)}
          ${run.reference_time_iso ? `· ref ${esc(run.reference_time_iso)}` : ''}
          · ${run.records.length} records</span>
        <span style="margin-left:auto"></span>
        <button class="btn small" id="replay-btn" title="re-derive every verdict from its record">replay all</button>
        <span id="replay-out" class="value-mono dim" style="margin-left:8px">
          ${run.replay ? `replay ${run.replay.checked - run.replay.wrong}/${run.replay.checked} wrong=${run.replay.wrong}` : ''}
        </span>
      </h2>
      ${run.chain_verified
        ? `<div class="chain-banner ok">🔒 chain verified — every record's digest links to its predecessor.</div>`
        : `<div class="chain-banner bad">✗ HASH CHAIN BROKEN — this sequence of records is not internally consistent.</div>`}
      ${run.kind === 'ledger' ? `<div class="note-box">ledger ${esc(run.path)} · streams: ${run.streams.map(esc).join(', ')}</div>` : ''}
    </div>
    ${sections.join('\n')}
    <div id="lab-out"></div>
  `;

  document.getElementById('replay-btn').addEventListener('click', async () => {
    try {
      const out = await api('/api/replay', {run_id: run.run_id});
      document.getElementById('replay-out').textContent =
        `replay ${out.checked - out.wrong}/${out.checked} wrong=${out.wrong}`;
    } catch (err) { alert(err.message); }
  });

  wireVerdictTable(run);
  if (run.waves) wirePipeline(run);
  wireSpine(run);
  if (Object.keys(run.labels || {}).length >= 4) wireCalibration(run);
  if (run.kind === 'certification') {
    renderCandidate(run, 0);
    $app.querySelectorAll('#cert-tabs .tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        $app.querySelectorAll('#cert-tabs .tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        renderCandidate(run, Number(tab.dataset.cand));
      });
    });
  }
  if (run.records.length && run.records.some((r) => r.raw_probabilities
        && Object.keys(r.raw_probabilities).length)) {
    S._lab = wireGateLab(run);
  }

  if (inspectorIdx !== null && inspectorIdx !== undefined && run.records[inspectorIdx]) {
    openInspector(run, inspectorIdx);
  }
}

function runBadge(run) {
  const names = {
    evaluate: 'Evaluation', certification: 'Model-swap certification',
    cascade: 'Cascade routing', recalibration: 'Nightly recalibration',
    ledger: 'Ledger',
  };
  return `<span class="chip RATIFIED" style="font-size:12px">${esc(names[run.kind] || run.kind)}</span>`;
}

/* --------------------------------------------------- verdict table */

function verdictSection(run, inspectorIdx) {
  const rows = run.rows.map((row) => {
    const flipped = S._lab && S._lab.flips && S._lab.flips.has(row.record_idx);
    const flip = flipped ? S._lab.flips.get(row.record_idx) : null;
    return `
    <tr class="clickable ${S.selected && S.selected.idx === row.record_idx ? 'selected' : ''}"
        data-idx="${row.record_idx}">
      <td class="value-mono dim">${esc(row.key)}</td>
      <td class="value-mono">${esc(row.question_id)}</td>
      <td>${chip(row.status)}${
        flip ? `<span class="flip-to faint">→</span> ${chip(flip.status, ' small')}` : ''}</td>
      <td class="value-mono">${row.value === null ? '<span class="faint">—</span>' : esc(fmtValue(row.value))}</td>
      <td><div class="conf-cell"><div class="conf-bar"><i style="width:${Math.round((row.confidence || 0) * 100)}%"></i></div><span class="conf-num">${fmt(row.confidence)}</span></div></td>
      <td class="value-mono dim">${fmt(row.top2_margin)}</td>
      <td class="value-mono dim">${fmt(row.normalized_entropy)}</td>
      <td class="value-mono faint">${esc(row.digest)}</td>
    </tr>`;
  }).join('');

  return `
  <div class="section" id="verdicts">
    <h2>Verdicts <span class="hint">fail-closed: a value appears only when RATIFIED · click a row</span></h2>
    <div class="card" style="padding:0">
      <table>
        <thead><tr>
          <th>case</th><th>question</th><th>status</th><th>value</th>
          <th>confidence</th><th>margin</th><th>entropy</th><th>digest</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

function wireVerdictTable(run) {
  const tbody = $app.querySelector('#verdicts tbody');
  tbody.querySelectorAll('tr').forEach((tr) => {
    tr.addEventListener('click', () => {
      const idx = Number(tr.dataset.idx);
      S.selected = {runId: run.run_id, idx};
      tbody.querySelectorAll('tr').forEach((x) => x.classList.remove('selected'));
      tr.classList.add('selected');
      openInspector(run, idx);
    });
  });
  document.onkeydown = (e) => {
    if (document.activeElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
    if (!S.selected || S.selected.runId !== run.run_id) return;
    const rows = run.rows;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const cur = rows.findIndex((r) => r.record_idx === S.selected.idx);
      const next = rows[Math.min(rows.length - 1, Math.max(0, cur + (e.key === 'ArrowDown' ? 1 : -1)))];
      if (next) {
        S.selected = {runId: run.run_id, idx: next.record_idx};
        const tr = tbody.querySelector(`tr[data-idx="${next.record_idx}"]`);
        tbody.querySelectorAll('tr').forEach((x) => x.classList.remove('selected'));
        if (tr) { tr.classList.add('selected'); tr.scrollIntoView({block: 'nearest'}); }
      }
    } else if (e.key === 'Enter') {
      openInspector(run, S.selected.idx);
    } else if (e.key === 'Escape') {
      closeDrawer();
    }
  };
}

/* --------------------------------------------------- inspector drawer */

function openInspector(run, idx) {
  const record = run.records[idx];
  if (!record) return;
  const verdict = record.verdict;
  const gi = record.gate_inputs;
  const policy = gi.policy;
  const compiled = record.compiled_question;
  const replayRow = (run.replay && run.replay.rows || []).find((r) => r.record_idx === idx);
  const label = (run.labels || {})[String(idx)];

  $drawerRoot.innerHTML = `
    <div class="scrim" id="scrim"></div>
    <div class="drawer" role="dialog" aria-label="verdict inspector">
      <h3>${esc(compiled.question_id)} ${chip(verdict.status)} ${replayRow
        ? `<span class="replay-badge ${replayRow.ok ? 'ok' : 'bad'}">${replayRow.ok ? 'replay ✓' : 'replay ✗'}</span>` : ''}
        <button class="close" id="drawer-close">esc</button></h3>
      <div class="sub">${esc(record.model_id)} · wave ${esc(record.wave_index)} · stream ${esc(record.stream_id)} · digest ${esc(record.record_digest.slice(0, 16))}…</div>

      ${label ? `<div class="note-box">pinned outcome (ground truth): <b class="mono">${esc(label)}</b>
        ${verdict.status === 'RATIFIED' && !valueMatches(verdict.value, label)
          ? '<span style="color:var(--red)">— this verdict disagrees with it</span>' : ''}</div>` : ''}

      <div class="section" style="margin:14px 0 0">
        <h2>Distribution <span class="hint">${esc(gi.primitive)}</span></h2>
        ${gi.primitive === 'choice' ? choiceChart(verdict, policy) : ''}
        ${gi.primitive === 'noul' ? noulChart(verdict, gi) : ''}
        ${gi.primitive === 'score' ? scoreChart(verdict, policy) : ''}
        ${gauges(verdict, gi)}
      </div>

      <div class="section" style="margin:14px 0 0">
        <h2>Question as the model saw it <span class="hint">keys are yours; only prose reached the sensor</span></h2>
        <div class="card">
          <div class="dim" style="font-size:12.5px">${esc(compiled.instructions)}</div>
          <div style="margin-top:8px">
            ${Object.entries(compiled.criteria).map(([k, v]) =>
              `<div class="dist-row" style="grid-template-columns:150px 1fr">
                 <div class="dist-label ${isEscape(k) ? 'escape' : ''}">${esc(k)}</div>
                 <div class="dim" style="font-size:11.5px">${esc(v)}</div>
               </div>`).join('')}
          </div>
        </div>
      </div>

      ${record.canonical_state && record.canonical_state['@deterministic_facts']
        ? `<div class="section" style="margin:14px 0 0">
             <h2>@deterministic_facts <span class="hint">resolved in native code, zero neural cost</span></h2>
             <div class="card value-mono" style="font-size:12px">
               ${Object.entries(record.canonical_state['@deterministic_facts']).map(([k, v]) =>
                 `<div>${esc(k)} = <b>${esc(JSON.stringify(v))}</b></div>`).join('') || '<span class="faint">none</span>'}
             </div>
           </div>` : ''}

      <details>
        <summary>raw audit record (s1core.audit/2)</summary>
        <pre>${esc(JSON.stringify(record, null, 2))}</pre>
      </details>
    </div>`;

  document.getElementById('scrim').addEventListener('click', closeDrawer);
  document.getElementById('drawer-close').addEventListener('click', closeDrawer);
}

function closeDrawer() {
  $drawerRoot.innerHTML = '';
}

function valueMatches(value, label) {
  if (typeof value === 'boolean') return (label === 'true') === value;
  return String(value) === String(label);
}

const ESCAPE_RE = /__abstain_/;
const isEscape = (k) => ESCAPE_RE.test(k);

function choiceChart(verdict, policy) {
  const probs = verdict.probabilities;
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const top1 = entries[0][0];
  const pct = (v) => `${Math.max(0.6, Math.round(v * 1000) / 10)}%`;
  const lines = entries.map(([key, p]) => `
    <div class="dist-row">
      <div class="dist-label ${isEscape(key) ? 'escape' : ''} ${key === top1 ? 'top1' : ''}">${esc(key)}${key === top1 ? ' ◀ top-1' : ''}</div>
      <div class="dist-bar">
        <i class="${key === top1 ? 'win' : ''} ${isEscape(key) ? 'escape' : ''}" style="width:${pct(p)}"></i>
        ${key === top1 && policy.min_top1_prob ? `<div class="dist-overlay" style="left:${Math.round(policy.min_top1_prob * 100)}%"><span class="tag">top1 ≥ ${policy.min_top1_prob}</span></div>` : ''}
      </div>
      <div class="dist-p">${fmt(p)}</div>
    </div>`).join('');
  const tauLine = policy.conformal_tau != null
    ? `<div class="note-box" style="margin-top:8px">conformal τ = <b class="mono">${fmt(policy.conformal_tau)}</b> → set size <b class="mono">${verdict.conformal_set_size}</b>${verdict.conformal_set_size !== 1 ? ' (must be exactly 1 to ratify)' : ''}</div>`
    : '';
  return `<div class="dist">${lines}</div>${tauLine}`;
}

function noulChart(verdict, gi) {
  const pTrue = verdict.probabilities.true ?? 0;
  const fe = gi.noul_false_edge, te = gi.noul_true_edge;
  const pos = (v) => `left:${Math.round(v * 1000) / 10}%`;
  return `
    <div class="noul-axis">
      <div class="noul-band" style="${pos(fe)}; width:${Math.round((te - fe) * 1000) / 10}%"></div>
      <div class="noul-marker" style="${pos(pTrue)}"><span class="val">p(true) = ${fmt(pTrue)}</span></div>
      <div class="noul-zone-label" style="${pos((fe + te) / 2)}">gray zone — fail closed</div>
    </div>
    <div class="noul-scale"><span>0 · false</span><span>p(true) · 1</span></div>
    <div class="note-box" style="margin-top:10px">edges: ratified <b class="mono">True</b> at p ≥ ${fmt(te, 2)} · ratified
      <b class="mono">False</b> at p ≤ ${fmt(fe, 2)} · between them the value is withheld, not rounded.</div>`;
}

function scoreChart(verdict, policy) {
  const probs = verdict.probabilities;
  const levels = Object.keys(probs).sort((a, b) => Number(a) - Number(b));
  const k = levels.length;
  const exp = verdict.raw_value;
  const W = 460, H = 170, PADL = 34, PADB = 30, PADT = 22;
  const inner = W - PADL - 16;
  const cx = (i) => PADL + (i + 0.5) * (inner / k);
  const maxP = Math.max(...Object.values(probs), 0.0001);
  const barH = (p) => (p / maxP) * (H - PADB - PADT);
  const expX = exp >= 1 && exp <= k
    ? PADL + ((exp - 1) / (k - 1)) * inner : null;
  const bars = levels.map((lvl, i) => {
    const h = barH(probs[lvl]);
    return `<rect x="${cx(i) - inner / k * 0.32}" y="${H - PADB - h}"
        width="${inner / k * 0.64}" height="${Math.max(1, h)}" rx="4"
        style="fill:var(--accent); opacity:0.85"></rect>
      <text x="${cx(i)}" y="${H - PADB - h - 5}" text-anchor="middle" font-size="10.5" style="fill:var(--muted); font-family:var(--mono)">${fmt(probs[lvl], 2)}</text>
      <text x="${cx(i)}" y="${H - 10}" text-anchor="middle" font-size="10.5" style="fill:var(--faint); font-family:var(--mono)">L${esc(lvl)}</text>`;
  }).join('');
  const marker = expX !== null ? `
    <line x1="${expX}" y1="${PADT - 8}" x2="${expX}" y2="${H - PADB}"
      style="stroke:var(--green)" stroke-width="2"/>
    <text x="${expX + 4}" y="${PADT - 1}" font-size="10.5" style="fill:var(--green); font-family:var(--mono)">s^ = ${fmt(exp, 2)} (expectation — an average, not the mode)</text>` : '';
  const modeI = levels.reduce((best, lvl, i) => probs[lvl] > probs[levels[best]] ? i : best, 0);
  const modeLine = `<line x1="${cx(modeI)}" y1="${PADT - 8}" x2="${cx(modeI)}" y2="${H - PADB}"
      style="stroke:var(--amber)" stroke-width="1.5" stroke-dasharray="4 3"/>
    <text x="${cx(modeI)}" y="${H - PADB + 20}" text-anchor="middle" font-size="9.5" style="fill:var(--amber); font-family:var(--mono)">mode L${esc(levels[modeI])} · AAD ${fmt(verdict.score_mode_deviation, 2)}</text>`;
  return `<svg width="100%" viewBox="0 0 ${W} ${H + 14}">
      <line x1="${PADL}" y1="${H - PADB}" x2="${W - 16}" y2="${H - PADB}" style="stroke:var(--border)"/>
      ${bars}${modeLine}${marker}
    </svg>
    <div class="note-box" style="margin-top:6px">rubric mass by level; the ratified value would be the expectation
      <b class="mono">s^ = ${fmt(exp, 3)}</b> — refused as BIMODAL_CONFLICT unless the distribution is tight enough
      (mode AAD ${fmt(verdict.score_mode_deviation)} vs max ${fmt(policy.max_score_mode_aad)}).</div>`;
}

function gauges(verdict, gi) {
  const policy = gi.policy;
  const g = (label, valueText, pass, thresholdText) => `
    <div class="gauge ${pass === null ? '' : pass ? 'pass' : 'fail'}">
      <div class="k">${esc(label)}</div>
      <div class="v">${esc(valueText)}</div>
      <div class="t">${esc(thresholdText)}${pass === null ? '' : pass ? ' ✓' : ' ✗'}</div>
    </div>`;
  const p = gi.primitive;
  const out = [];
  out.push(g('confidence', fmt(verdict.confidence), null, '1 − uncertainty'));
  if (p === 'choice') {
    out.push(g('top-1 prob', fmt(top1Of(verdict)), top1Of(verdict) >= policy.min_top1_prob, `≥ ${policy.min_top1_prob}`));
    out.push(g('top-2 margin', fmt(verdict.top2_margin), verdict.top2_margin >= policy.min_top2_margin, `≥ ${policy.min_top2_margin}`));
    out.push(g('entropy', fmt(verdict.normalized_entropy), verdict.normalized_entropy <= policy.max_normalized_entropy, `≤ ${policy.max_normalized_entropy}`));
  }
  if (p === 'score') {
    out.push(g('mode AAD', fmt(verdict.score_mode_deviation), verdict.score_mode_deviation <= policy.max_score_mode_aad, `≤ ${policy.max_score_mode_aad}`));
  }
  if (p === 'noul') {
    const inGray = verdict.status === 'ABSTAINED_GATE';
    out.push(g('edge position', inGray ? 'in gray zone' : 'past an edge', !inGray, '0.35 / 0.75'));
  }
  if (verdict.permutation_divergence != null) {
    out.push(g('permutation JS', fmt(verdict.permutation_divergence),
      verdict.permutation_divergence <= policy.max_permutation_divergence,
      `≤ ${policy.max_permutation_divergence} bits`));
  }
  out.push(g('temperature', fmt(policy.temperature), null, 'calibration T'));
  return `<div class="gauges">${out.join('')}</div>`;
}

function top1Of(verdict) {
  const vals = Object.values(verdict.probabilities);
  return vals.length ? Math.max(...vals) : 0;
}

/* --------------------------------------------------- pipeline / waves */

function pipelineSection(run) {
  const meta = Object.fromEntries((run.questions_meta || []).map((q) => [q.question_id, q]));
  const waves = run.waves;
  const byQ = {};
  for (const row of run.rows) if (!(row.question_id in byQ)) byQ[row.question_id] = row;

  const COL_W = 190, ROW_H = 64, PAD = 26, GAP = 92;
  const rowsPerWave = waves.map((w) => w.length);
  const height = Math.max(...rowsPerWave) * ROW_H + PAD * 2 + 26;
  const width = waves.length * (COL_W + GAP) + PAD * 2 - GAP;

  const posOf = {};
  const nodes = waves.map((wave, wi) => wave.map((qid, qi) => {
    const x = PAD + wi * (COL_W + GAP);
    const y = PAD + 26 + qi * ROW_H;
    posOf[qid] = {x, y};
    const row = byQ[qid] || {};
    const cls = STATUS_CLASS[row.status] || 'SKIPPED_DEPENDENCY';
    const sub = (row.status || '')
      + (row.value !== null && row.value !== undefined
        ? ' · ' + fmtValue(row.value).slice(0, 14) : '');
    return `<g class="node" data-qid="${esc(qid)}" style="cursor:pointer">
      <rect x="${x}" y="${y}" width="${COL_W}" height="46" rx="9"
        style="fill:var(--panel-2); stroke:${strokeFor(cls)}" stroke-width="1.5"/>
      <text x="${x + 12}" y="${y + 19}" style="fill:var(--text)" font-size="12.5" font-weight="600">${esc(qid)}</text>
      <text x="${x + 12}" y="${y + 35}" style="fill:${strokeFor(cls)}" font-size="10.5">${esc(sub)}</text>
    </g>`;
  }).join('')).join('');

  const edges = [];
  const edgeLabels = [];
  for (const [qid, q] of Object.entries(meta)) {
    for (const [parent, required] of (q.depends_on || [])) {
      const a = posOf[parent], b = posOf[qid];
      if (!a || !b) continue;
      const x1 = a.x + COL_W, y1 = a.y + 23, x2 = b.x, y2 = b.y + 23;
      const mid = (x1 + x2) / 2;
      edges.push(`<path d="M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}"
        style="fill:none; stroke:var(--border)" stroke-width="1.5"/>`);
      // labels paint in their own layer above the nodes so nothing clips them
      edgeLabels.push(`<text x="${mid}" y="${y2 - 8}" style="fill:var(--faint); stroke:var(--bg); stroke-width:3px; paint-order:stroke" font-size="9.5" text-anchor="middle">requires ${esc(required)}</text>`);
    }
  }

  const waveLabels = waves.map((wave, wi) => {
    const x = PAD + wi * (COL_W + GAP);
    return `<text x="${x}" y="${PAD + 8}" style="fill:var(--faint)" font-size="10">WAVE ${wi}</text>`;
  }).join('');

  return `
  <div class="section" id="pipeline">
    <h2>Pipeline <span class="hint">topological waves — a child runs only where its parents ratified; color is where the decision stopped</span></h2>
    <div class="card">
      <div class="stage-strip">
        <span class="stage-chip">raw state</span><span class="stage-arrow">→</span>
        <span class="stage-chip">@deterministic_facts</span><span class="stage-arrow">→</span>
        <span class="stage-chip">waves (${waves.length})</span><span class="stage-arrow">→</span>
        <span class="stage-chip">admissibility gates</span><span class="stage-arrow">→</span>
        <span class="stage-chip">audit ledger</span>
      </div>
      <div class="pipeline-wrap"><svg class="pipeline" width="${width}" height="${height}">
        ${edges}${nodes}${waveLabels}${edgeLabels.join('')}
      </svg></div>
    </div>
  </div>`;
}

function strokeFor(cls) {
  return {
    RATIFIED: 'var(--green)', ABSTAINED_MODEL: 'var(--amber)',
    ABSTAINED_GATE: 'var(--amber)', BIMODAL_CONFLICT: 'var(--orange)',
    SKIPPED_DEPENDENCY: 'var(--gray)', LOOP_GUARD_TRIPPED: 'var(--red)',
  }[cls] || 'var(--gray)';
}

function wirePipeline(run) {
  const byQ = {};
  for (const row of run.rows) if (!(row.question_id in byQ)) byQ[row.question_id] = row;
  $app.querySelectorAll('#pipeline .node').forEach((node) => {
    node.addEventListener('click', () => {
      const row = byQ[node.dataset.qid];
      if (row) openInspector(run, row.record_idx);
    });
  });
}

/* --------------------------------------------------- ledger spine */

function ledgerSection(run) {
  const streams = [...new Set(run.rows.map((r) => r.key))];
  const filter = streams.length > 1
    ? `<select id="stream-filter" style="margin-left:10px"><option value="">all streams (${streams.length})</option>
        ${streams.map((s) => `<option>${esc(s)}</option>`).join('')}</select>`
    : '';
  const rows = run.records.map((record, idx) => {
    const verdict = record.verdict;
    const ok = run.replay.rows[idx] ? run.replay.rows[idx].ok : null;
    return `<div class="spine-row ${ok === false ? 'link-bad' : 'link-ok'}" data-idx="${idx}" data-stream="${esc(record.stream_id)}">
      <span class="spine-digest">${esc((record.record_digest || '').slice(0, 10))}…</span>
      <span class="value-mono dim">${esc((record.compiled_question || {}).question_id || '?')}</span>
      ${chip(verdict.status, ' small')}
      <span class="value-mono">${verdict.value === null ? '' : esc(JSON.stringify(verdict.value))}</span>
      <span class="replay-badge ${ok === false ? 'bad' : 'ok'}" style="margin-left:auto">${ok === false ? 'replay ✗' : 'replay ✓'}</span>
    </div>`;
  }).join('');

  return `
  <div class="section" id="ledger">
    <h2>Ledger spine <span class="hint">hash-chained · click any record · filter by stream</span>${filter}</h2>
    <div class="card"><div class="spine">${rows}</div></div>
  </div>`;
}

function wireSpine(run) {
  const filter = document.getElementById('stream-filter');
  const applyFilter = () => {
    const want = filter ? filter.value : '';
    $app.querySelectorAll('.spine-row').forEach((row) => {
      row.style.display = !want || row.dataset.stream === want ? '' : 'none';
    });
  };
  if (filter) filter.addEventListener('change', applyFilter);
  $app.querySelectorAll('.spine-row').forEach((row) => {
    row.addEventListener('click', () => openInspector(run, Number(row.dataset.idx)));
  });
}

/* --------------------------------------------------- gate lab */

function gateLabSection(run) {
  const defaults = {
    min_top1_prob: 0.65, min_top2_margin: 0.20, max_normalized_entropy: 0.55,
    max_score_mode_aad: 0.25, temperature: 1.0, conformal_tau: null,
  };
  const dials = [
    ['min_top1_prob', 'top-1 floor', 0, 1, 0.01],
    ['min_top2_margin', 'top-2 margin', 0, 1, 0.01],
    ['max_normalized_entropy', 'max entropy', 0, 1, 0.01],
    ['max_score_mode_aad', 'max score AAD', 0, 1, 0.01],
    ['temperature', 'temperature', 0.1, 3, 0.01],
  ].map(([key, label, min, max, step]) => `
    <div class="dial">
      <label>${label}</label>
      <input type="range" min="${min}" max="${max}" step="${step}" value="${defaults[key]}" data-key="${key}">
      <span class="num" id="num-${key}">${defaults[key]}</span>
    </div>`).join('') + `
    <div class="dial">
      <label>conformal τ</label>
      <input type="range" min="0" max="1" step="0.01" value="0.5" data-key="conformal_tau" id="tau-range">
      <span class="num" id="num-conformal_tau">off</span>
    </div>
    <div class="dial"><label></label>
      <div><label style="text-transform:none;letter-spacing:0"><input type="checkbox" id="tau-enable"> apply τ gate (choice only)</label></div><span></span>
    </div>`;

  return `
  <div class="section" id="gate-lab">
    <h2>Gate lab <span class="hint">the dials re-run the pure Stage-4 gate over this run's stored raw distributions — server-side, s1core itself</span></h2>
    <div class="card">
      ${dials}
      <div style="display:flex; gap:8px; margin-top:10px; align-items:center">
        <button class="btn small" id="lab-reset">reset to defaults</button>
        <span class="dim" id="lab-summary" style="font-size:12.5px"></span>
      </div>
      <div class="flip-strip" id="flip-strip"></div>
      <div class="note-box" style="margin-top:12px">Dial outcomes are what the harness <i>would</i> release under that
        policy — this is the same admissibility gate, applied to the same raw distributions. RATIFIED still means a
        policy passed, not that an answer is correct.</div>
    </div>
  </div>`;
}

function wireGateLab(run) {
  const lab = {timer: null, flips: new Map(), current: {...gateDefaults()}};
  const readPolicy = () => {
    const policy = {};
    $app.querySelectorAll('#gate-lab input[type="range"]').forEach((r) => {
      const key = r.dataset.key;
      if (key === 'conformal_tau') {
        policy[key] = document.getElementById('tau-enable').checked ? Number(r.value) : null;
      } else {
        policy[key] = Number(r.value);
      }
    });
    return policy;
  };

  const sweep = async () => {
    const policy = readPolicy();
    try {
      const out = await api('/api/sweep', {run_id: run.run_id, policy});
      lab.flips = new Map(out.rows.filter((r) => r.changed).map((r) => [r.record_idx, r]));
      const ratified = out.rows.filter((r) => r.status === 'RATIFIED').length;
      const baseRat = out.rows.filter((r) => r.base_status === 'RATIFIED').length;
      document.getElementById('lab-summary').textContent =
        `ratified ${baseRat} → ${ratified} of ${out.rows.length} · ${out.rows.filter((r) => r.changed).length} verdicts flip`;
      const flips = out.rows.filter((r) => r.changed);
      document.getElementById('flip-strip').innerHTML = flips.length
        ? flips.slice(0, 24).map((r) => `<div class="flip-row">
            <span class="dim">${esc(r.question_id)}</span>
            ${chip(r.base_status, ' small')} <span class="faint">→</span> ${chip(r.status, ' small')}
            ${r.value !== null ? `<span class="value-mono">${esc(JSON.stringify(r.value))}</span>` : ''}
          </div>`).join('') + (flips.length > 24 ? `<div class="faint" style="font-size:11.5px">… ${flips.length - 24} more</div>` : '')
        : '<div class="faint" style="font-size:12px">no verdicts flip under this policy.</div>';
      // annotate verdict table
      $app.querySelectorAll('#verdicts tbody tr').forEach((tr) => {
        const idx = Number(tr.dataset.idx);
        const flip = lab.flips.get(idx);
        const cell = tr.children[2];
        const old = cell.innerHTML.split('<span class="flip-to"')[0];
        cell.innerHTML = flip ? `${old}<span class="flip-to faint">→</span> ${chip(flip.status, ' small')}` : old;
      });
    } catch (err) {
      document.getElementById('lab-summary').textContent = err.message;
    }
  };

  let timer = null;
  $app.querySelectorAll('#gate-lab input[type="range"]').forEach((r) => {
    r.addEventListener('input', () => {
      if (r.dataset.key !== 'conformal_tau') {
        document.getElementById(`num-${r.dataset.key}`).textContent = r.value;
      } else {
        document.getElementById('num-conformal_tau').textContent =
          document.getElementById('tau-enable').checked ? r.value : 'off';
      }
      clearTimeout(timer);
      timer = setTimeout(sweep, 150);
    });
  });
  document.getElementById('tau-enable').addEventListener('change', sweep);
  document.getElementById('lab-reset').addEventListener('click', () => {
    $app.querySelectorAll('#gate-lab input[type="range"]').forEach((r) => {
      r.value = gateDefaults()[r.dataset.key] ?? 0.5;
      r.dispatchEvent(new Event('input'));
    });
    document.getElementById('tau-enable').checked = false;
    sweep();
  });
  return lab;
}

function gateDefaults() {
  return {
    min_top1_prob: 0.65, min_top2_margin: 0.20, max_normalized_entropy: 0.55,
    max_score_mode_aad: 0.25, temperature: 1.0, conformal_tau: 0.5,
  };
}

/* --------------------------------------------------- calibration panel */

function calibrationSection(run) {
  const qids = [...new Set(Object.keys(run.labels).map((k) => run.rows.find((r) => r.record_idx === Number(k))?.question_id).filter(Boolean))];
  return `
  <div class="section" id="calibration">
    <h2>Calibration <span class="hint">temperature on the older 60%, conformal τ on the newer 40% — disjoint, as the fit requires</span></h2>
    <div class="card">
      <div class="kv">
        <div>
          <label class="fld">question</label>
          <select id="cal-q">${qids.map((q) => `<option>${esc(q)}</option>`).join('')}</select>
        </div>
        <div>
          <label class="fld">risk α</label>
          <input type="number" id="cal-alpha" min="0.01" max="0.5" step="0.01" value="0.10">
        </div>
      </div>
      <div style="margin-top:10px"><button class="btn primary" id="cal-fit">Fit from labels</button>
        <span class="dim" id="cal-n" style="margin-left:8px; font-size:12px"></span></div>
      <div id="cal-out"></div>
    </div>
  </div>`;
}

function wireCalibration(run) {
  document.getElementById('cal-fit').addEventListener('click', async () => {
    const q = document.getElementById('cal-q').value;
    const alpha = Number(document.getElementById('cal-alpha').value);
    const nLabeled = Object.keys(run.labels).filter((k) => {
      const row = run.rows.find((r) => r.record_idx === Number(k));
      return row && row.question_id === q;
    }).length;
    if (nLabeled < 4) {
      document.getElementById('cal-out').innerHTML =
        `<div class="error-box">only ${nLabeled} labeled samples for ${esc(q)} — need at least 4.</div>`;
      return;
    }
    try {
      const fit = await api('/api/calibrate', {run_id: run.run_id, question_id: q, alpha});
      renderFit(q, fit);
    } catch (err) {
      document.getElementById('cal-out').innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
    }
  });
}

function renderFit(q, fit) {
  const eceW = (v) => Math.min(100, Math.round(v * 250));
  const bins = fit.bins || [];
  const W = 260, H = 150, PAD = 24;
  const sx = (v) => PAD + v * (W - PAD * 2);
  const sy = (v) => H - PAD - v * (H - PAD * 2);
  const maxN = Math.max(1, ...bins.map((b) => b.n));
  const bars = bins.map((b, i) => {
    const x = sx(b.conf) - 9, w = 18, h = (b.n / maxN) * (H - PAD * 2) * 0.9;
    return `<rect x="${x}" y="${H - PAD - h}" width="${w}" height="${h}" style="fill:var(--border)" opacity="0.8"/>`;
  }).join('');
  const dots = bins.filter((b) => b.n).map((b) =>
    `<circle cx="${sx(b.conf)}" cy="${sy(b.acc)}" r="4" style="fill:var(--accent)"/>
     <line x1="${sx(b.conf)}" y1="${sy(b.acc)}" x2="${sx(b.conf)}" y2="${sy(b.conf)}" style="stroke:var(--red)" stroke-width="1" opacity="0.6"/>`).join('');
  const diag = `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" style="stroke:var(--faint)" stroke-dasharray="3 3"/>`;

  document.getElementById('cal-out').innerHTML = `
    <div class="rel-wrap" style="margin-top:14px">
      <svg width="${W}" height="${H}">
        ${bars}${diag}${dots}
        <text x="${sx(0)}" y="${H - 6}" style="fill:var(--faint); font-family:var(--mono)" font-size="9">0</text>
        <text x="${sx(1)}" y="${H - 6}" style="fill:var(--faint); font-family:var(--mono)" font-size="9" text-anchor="end">conf 1.0</text>
      </svg>
      <div class="rel-legend">
        <div><b>${esc(q)}</b> (${esc(fit.primitive)}) · n = ${fit.n_fit}+${fit.n_cal} (temp+conformal)</div>
        <div>temperature <b>${fmt(fit.temperature)}</b>${fit.conformal_tau != null ? ` · conformal τ <b>${fmt(fit.conformal_tau)}</b> at α=${fit.alpha}` : ' · τ n/a for this primitive'}</div>
        <div>ECE <b>${fmt(fit.ece_before)}</b> → <b style="color:var(--green)">${fmt(fit.ece_after)}</b></div>
        ${fit.coverage !== null && fit.coverage !== undefined ? `<div>measured coverage <b>${fmt(fit.coverage)}</b> ≥ 1−α (marginal, exchangeability assumed)</div>` : ''}
        <div class="faint">gray bars: sample mass per confidence bin · blue: accuracy · red stub: gap to confidence</div>
      </div>
    </div>`;
}

/* --------------------------------------------------- certification view */

function certificationSection(run) {
  const tabs = run.candidates.map((c, i) =>
    `<button class="tab ${i === 0 ? 'active' : ''}" data-cand="${i}">${esc(c.model_id)} — ${c.promoted ? 'PROMOTE' : 'REJECT'}</button>`).join('');
  return `
  <div class="section" id="certification">
    <h2>Certification <span class="hint">the only law that promotes is wrong == 0 · abstention is coverage loss, not wrongness</span></h2>
    <div class="tabs" id="cert-tabs">${tabs}</div>
    <div id="cert-body"></div>
  </div>`;
}

function renderCandidate(run, i) {
  const c = run.candidates[i];
  const banner = c.promoted
    ? `<div class="cert-banner promote"><span class="verdict">CERTIFICATE: PROMOTE</span> — wrong=0 on ${run.pinned_cases} pinned case-questions; abstentions are coverage loss, not wrongness.</div>`
    : `<div class="cert-banner reject"><span class="verdict">CERTIFICATE: REJECT</span> — wrong=${c.wrong_count} (must be 0): ${c.wrongs.map((w) => `${esc(w.question_id)}/${esc(w.case_id)} ratified ${esc(JSON.stringify(w.ratified))}, pinned ${esc(JSON.stringify(w.pinned))}`).join('; ')}</div>`;
  const rows = c.rows.map((r) => {
    const heat = Math.min(1, r.drift * 2.5);
    const heatColor = `rgba(248, 81, 73, ${Math.round(heat * 60) / 100})`;
    return `<tr>
      <td class="value-mono dim">${esc(r.case_id)}</td>
      <td class="value-mono">${esc(r.question_id)}</td>
      <td>${chip(r.base_status, ' small')}</td>
      <td class="value-mono">${r.base_value === null ? '—' : esc(JSON.stringify(r.base_value))}</td>
      <td>${chip(r.cand_status, ' small')}</td>
      <td class="value-mono">${r.cand_value === null ? '—' : esc(JSON.stringify(r.cand_value))}${r.wrong ? ' <span class="chip LOOP_GUARD_TRIPPED small">WRONG</span>' : ''}</td>
      <td><span class="heat" style="background:${heatColor}">${fmt(r.drift, 3)}</span></td>
      <td class="value-mono dim">${esc(JSON.stringify(r.pinned))}</td>
    </tr>`;
  }).join('');
  const pq = Object.entries(c.per_question).map(([qid, s]) =>
    `<span class="stage-chip">${esc(qid)}: wrong ${s.wrong} · ratified ${s.ratified} · drift ${fmt(s.mean_drift)}–${fmt(s.max_drift)} bits</span>`).join('<span class="stage-arrow">·</span> ');
  document.getElementById('cert-body').innerHTML = `
    ${banner}
    <div class="stage-strip" style="margin-bottom:10px">${pq}</div>
    <div class="card" style="padding:0">
      <table>
        <thead><tr><th>case</th><th>question</th><th>baseline</th><th>value</th><th>candidate</th><th>value</th><th>JS drift (bits)</th><th>pinned</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* --------------------------------------------------- cascade view */

function cascadeSection(run) {
  const cheapest = run.cheapest_zero_wrong;
  const rows = run.tiers.map((t) => `
    <tr class="${t.name === cheapest ? 'selected' : ''}">
      <td class="value-mono">${esc(t.name)}${t.name === cheapest ? ' ◀ cheapest zero-wrong' : ''}</td>
      <td class="value-mono dim" title="${esc(JSON.stringify(t.policy))}">T=${fmt(t.policy.temperature, 2)} top1≥${t.policy.min_top1_prob}${t.policy.conformal_tau != null ? ` τ=${fmt(t.policy.conformal_tau, 2)}` : ''}</td>
      <td class="value-mono">${t.handled} <span class="faint">(${Math.round((t.handled / run.routed_cases) * 100)}%)</span></td>
      <td class="value-mono" style="${t.wrong ? 'color:var(--red)' : 'color:var(--green)'}">${t.wrong}</td>
      <td class="value-mono">${t.escalated}</td>
      <td class="value-mono">${t.avg_latency_ms} ms</td>
      <td class="value-mono">$${fmt(t.cost_per_1k, 4)}</td>
    </tr>`).join('');
  return `
  <div class="section" id="tiers">
    <h2>Cascade tiers <span class="hint">the admissibility gate as a cost dial — RATIFIED ships the ~110 ms answer, the rest escalate</span></h2>
    <div class="card" style="padding:0">
      <table>
        <thead><tr><th>tier</th><th>policy</th><th>S1 handled</th><th>shipped wrong</th><th>escalated</th><th>avg latency</th><th>$ / 1k cases</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="note-box">fitted tier: T=${fmt(run.fitted.temperature)} · τ=${fmt(run.fitted.conformal_tau)} from a disjoint
      ${run.calibration_window}-case window. Escalated cases are modeled as correctly resolved by System 2 — measure its
      error rate the same way, from its own ledger.</div>
  </div>`;
}

/* --------------------------------------------------- recalibration view */

function recalibrationSection(run) {
  const cards = Object.entries(run.fits).map(([qid, f]) => `
    <div class="card">
      <h3>${esc(qid)} <span class="badge-det">${esc(f.primitive)}</span></h3>
      <div class="gauges">
        <div class="gauge"><div class="k">temperature</div><div class="v">${fmt(f.temperature)}</div><div class="t">1.0 raw</div></div>
        ${f.conformal_tau != null ? `<div class="gauge"><div class="k">conformal τ</div><div class="v">${fmt(f.conformal_tau)}</div><div class="t">α=${f.alpha}</div></div>` : ''}
        <div class="gauge pass"><div class="k">ECE</div><div class="v">${fmt(f.ece_before)} → ${fmt(f.ece_after)}</div><div class="t">raw → fitted ✓</div></div>
        ${f.coverage != null ? `<div class="gauge ${f.coverage >= 1 - f.alpha ? 'pass' : 'fail'}"><div class="k">coverage</div><div class="v">${fmt(f.coverage)}</div><div class="t">≥ 1−α on cal slice</div></div>` : ''}
      </div>
    </div>`).join('');
  const cfRows = Object.entries(run.counterfactual).map(([qid, c]) => `
    <tr>
      <td class="value-mono">${esc(qid)}</td>
      <td class="value-mono">${c.def_rat}/${c.n} → <b>${c.fit_rat}/${c.n}</b></td>
      <td class="value-mono">${c.def_wrong} → <b>${c.fit_wrong}</b></td>
    </tr>`).join('');
  return `
  <div class="section" id="recal">
    <h2>Fits <span class="hint">snapshot digest ${esc(run.snapshot_digest.slice(0, 16))}… · ${run.labeled_cases}/${run.window_cases} labeled cases</span></h2>
    <div class="grid cols-3">${cards}</div>
    <div class="card" style="margin-top:12px; padding:0">
      <table>
        <thead><tr><th>question</th><th>ratified (default → fitted)</th><th>wrong-while-ratified</th></tr></thead>
        <tbody>${cfRows}</tbody>
      </table>
    </div>
    <div class="note-box">Reading: temperature recovers the automation the default gate refused; residual wrongness is the
      sensor's confusion rate living inside the α budget — calibration <i>prices</i> errors, it does not cure them. Cure
      paths: certify a better sensor (model-swap) and escalate what the gate refuses (cascade).</div>
  </div>`;
}

/* --------------------------------------------------- composer */

function defaultComposer() {
  return {
    rawState: '{\n  "ticket_body": "Customer says the refund they requested never arrived and rent is due tomorrow",\n  "order_total": 120.0,\n  "requested_refund": 150.0\n}',
    reference: '2026-10-15T12:00:00+00:00',
    streamId: 'ui-composer',
    backend: 'auto',
    questions: [
      {
        type: 'choice', question_id: 'refund_lane',
        instructions: 'Which fulfillment lane should handle this refund request?',
        options: [
          {k: 'auto_approve', v: 'Refund is within policy limits and history supports automatic approval.'},
          {k: 'agent_review', v: 'A human agent must review the refund amount before approval.'},
          {k: 'deny', v: 'The request falls outside the refund policy and must be denied.'},
        ],
        dependsOn: '',
      },
      {
        type: 'noul', question_id: 'exceeds_total',
        instructions: 'Does the requested refund exceed the full order total?',
        trueCriteria: 'The requested refund is greater than the order total.',
        falseCriteria: 'The requested refund is within the order total.',
        dependsOn: 'refund_lane = agent_review',
      },
    ],
  };
}

function renderComposer() {
  closeDrawer();
  if (!S.composer) S.composer = defaultComposer();
  const c = S.composer;

  $app.innerHTML = `
    <div class="hero"><h1>Composer</h1>
      <p>Declare questions — full-sentence instructions and criteria (terse prose is a compile error, on purpose;
      only prose ever reaches the model). Evaluate against the toy sensor offline, or the live model when a key is
      configured.</p></div>
    <div class="grid" style="grid-template-columns: 1.1fr 1fr; align-items:start">
      <div class="section" style="margin-top:0">
        <h2>Questions</h2>
        <div id="q-list"></div>
        <div style="display:flex; gap:6px; margin-top:8px">
          <button class="btn" id="add-choice">+ choice</button>
          <button class="btn" id="add-score">+ score</button>
          <button class="btn" id="add-noul">+ noul</button>
        </div>
      </div>
      <div class="section" style="margin-top:0">
        <h2>State</h2>
        <div class="card">
          <label class="fld">raw state (JSON)</label>
          <textarea id="c-state" spellcheck="false">${esc(c.rawState)}</textarea>
          <div class="kv" style="margin-top:6px">
            <div><label class="fld">reference time (ISO)</label><input type="text" id="c-ref" value="${esc(c.reference)}"></div>
            <div><label class="fld">stream id</label><input type="text" id="c-stream" value="${esc(c.streamId)}"></div>
          </div>
          <label class="fld">backend</label>
          <select id="c-backend">
            <option value="auto"${c.backend === 'auto' ? ' selected' : ''}>auto (live if key configured)</option>
            <option value="toy"${c.backend === 'toy' ? ' selected' : ''}>toy (deterministic lexical)</option>
            ${S.state && S.state.key_present ? `<option value="live"${c.backend === 'live' ? ' selected' : ''}>live (${esc(S.state.backend.model_id)})</option>` : ''}
          </select>
          <div style="margin-top:14px"><button class="btn primary" id="c-run">Evaluate →</button>
            <span class="dim" style="margin-left:8px; font-size:12px" id="c-hint">↑↓/Enter navigate results · esc closes the inspector</span></div>
          <div id="c-error"></div>
        </div>
      </div>
    </div>`;

  renderComposerQuestions();

  const bind = (id, key) => {
    document.getElementById(id).addEventListener('input', (e) => {
      S.composer[key] = e.target.value;
    });
  };
  bind('c-state', 'rawState'); bind('c-ref', 'reference');
  bind('c-stream', 'streamId');
  document.getElementById('c-backend').addEventListener('change', (e) => {
    S.composer.backend = e.target.value;
  });
  document.getElementById('add-choice').addEventListener('click', () => addQuestion('choice'));
  document.getElementById('add-score').addEventListener('click', () => addQuestion('score'));
  document.getElementById('add-noul').addEventListener('click', () => addQuestion('noul'));
  document.getElementById('c-run').addEventListener('click', runComposer);
}

function renderComposerQuestions() {
  const c = S.composer;
  const list = document.getElementById('q-list');
  list.innerHTML = c.questions.map((q, i) => `
    <div class="q-card" data-qi="${i}">
      <div class="q-head">
        <span class="chip ABSTAINED_GATE small">${esc(q.type)}</span>
        <span class="qid">${esc(q.question_id)}</span>
        <span style="margin-left:auto"></span>
        <button class="btn small" data-act="up">↑</button>
        <button class="btn small" data-act="down">↓</button>
        <button class="btn small" data-act="del">✕</button>
      </div>
      <div class="kv">
        <div><label class="fld">question_id (your key)</label><input type="text" data-fld="question_id" value="${esc(q.question_id)}"></div>
        <div><label class="fld">depends_on (parent = value, blank for none)</label><input type="text" data-fld="dependsOn" value="${esc(q.dependsOn || '')}" placeholder="refund_lane = agent_review"></div>
      </div>
      <label class="fld">instructions (≥ 4 words — the compile lint enforces it)</label>
      <textarea data-fld="instructions" style="min-height:44px">${esc(q.instructions)}</textarea>
      ${q.type === 'choice' ? `
        <label class="fld">options (key → criteria prose, ≥ 3 words each; escape hatches are injected automatically)</label>
        ${q.options.map((o, oi) => `
          <div class="option-row">
            <input type="text" data-opt="${oi}" data-ofld="k" value="${esc(o.k)}">
            <input type="text" data-opt="${oi}" data-ofld="v" value="${esc(o.v)}">
          </div>`).join('')}
        <button class="btn small" data-act="addopt">+ option</button>` : ''}
      ${q.type === 'score' ? `
        <label class="fld">rubric levels 1..k (2–10 contiguous)</label>
        ${q.levels.map((o, oi) => `
          <div class="option-row" style="grid-template-columns:40px 1fr">
            <input type="text" data-opt="${oi}" data-ofld="k" value="${esc(o.k)}" disabled style="opacity:0.7">
            <input type="text" data-opt="${oi}" data-ofld="v" value="${esc(o.v)}">
          </div>`).join('')}` : ''}
      ${q.type === 'noul' ? `
        <label class="fld">true criteria</label><textarea data-fld="trueCriteria" style="min-height:36px">${esc(q.trueCriteria)}</textarea>
        <label class="fld">false criteria</label><textarea data-fld="falseCriteria" style="min-height:36px">${esc(q.falseCriteria)}</textarea>` : ''}
    </div>`).join('');

  list.querySelectorAll('.q-card').forEach((card) => {
    const i = Number(card.dataset.qi);
    card.querySelectorAll('[data-fld]').forEach((inp) => {
      inp.addEventListener('input', () => {
        S.composer.questions[i][inp.dataset.fld] = inp.value;
      });
    });
    card.querySelectorAll('[data-opt]').forEach((inp) => {
      inp.addEventListener('input', () => {
        const q = S.composer.questions[i];
        (q.options || q.levels)[Number(inp.dataset.opt)][inp.dataset.ofld] = inp.value;
      });
    });
    const act = (name, fn) => card.querySelector(`[data-act="${name}"]`).addEventListener('click', fn);
    act('del', () => { S.composer.questions.splice(i, 1); renderComposerQuestions(); });
    act('up', () => { if (i > 0) { [S.composer.questions[i - 1], S.composer.questions[i]] = [S.composer.questions[i], S.composer.questions[i - 1]]; renderComposerQuestions(); } });
    act('down', () => { if (i < S.composer.questions.length - 1) { [S.composer.questions[i + 1], S.composer.questions[i]] = [S.composer.questions[i], S.composer.questions[i + 1]]; renderComposerQuestions(); } });
    if (card.querySelector('[data-act="addopt"]')) {
      act('addopt', () => {
        S.composer.questions[i].options.push({k: `option_${S.composer.questions[i].options.length + 1}`, v: 'Describe what makes this option the right choice.'});
        renderComposerQuestions();
      });
    }
  });
}

function addQuestion(type) {
  const base = {type, question_id: `q_${type}_${S.composer.questions.length + 1}`, dependsOn: ''};
  if (type === 'choice') Object.assign(base, {
    instructions: 'Describe the decision this question should make in a full sentence.',
    options: [
      {k: 'first_option', v: 'Describe what makes the first option the right choice.'},
      {k: 'second_option', v: 'Describe what makes the second option the right choice.'},
    ],
  });
  if (type === 'score') Object.assign(base, {
    instructions: 'Rate this state on the rubric from low to high.',
    levels: [
      {k: 1, v: 'The lowest rubric level, described precisely.'},
      {k: 2, v: 'A middle rubric level, described precisely.'},
      {k: 3, v: 'The highest rubric level, described precisely.'},
    ],
  });
  if (type === 'noul') Object.assign(base, {
    instructions: 'Ask one binary proposition about the state in a full sentence.',
    trueCriteria: 'The proposition holds, stated as checkable criteria.',
    falseCriteria: 'The proposition fails, stated as checkable criteria.',
  });
  S.composer.questions.push(base);
  renderComposerQuestions();
}

async function runComposer() {
  const c = S.composer;
  const errBox = document.getElementById('c-error');
  errBox.innerHTML = '';
  let rawState;
  try {
    rawState = JSON.parse(c.rawState);
  } catch (e) {
    errBox.innerHTML = `<div class="error-box">raw state is not valid JSON: ${esc(e.message)}</div>`;
    return;
  }
  let questions;
  try {
    questions = c.questions.map((q) => {
      const out = {type: q.type, question_id: q.question_id.trim(), instructions: q.instructions};
      if (q.dependsOn && q.dependsOn.trim()) {
        const [parent, value] = q.dependsOn.split('=');
        out.depends_on = [parent.trim(), (value || '').trim()];
      }
      if (q.type === 'choice') out.options = Object.fromEntries(q.options.map((o) => [o.k, o.v]));
      if (q.type === 'score') out.rubric_levels = Object.fromEntries(q.levels.map((o) => [Number(o.k), o.v]));
      if (q.type === 'noul') {
        out.true_criteria = q.trueCriteria;
        out.false_criteria = q.falseCriteria;
      }
      return out;
    });
  } catch (e) {
    errBox.innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
    return;
  }
  try {
    const payload = await api('/api/evaluate', {
      raw_state: rawState,
      questions,
      backend: c.backend,
      reference_time_iso: c.reference || null,
      stream_id: c.streamId || 'ui-composer',
    });
    rememberRun(payload, `composer · ${payload.verdicts ? Object.keys(payload.verdicts).length : payload.rows.length} questions`);
    nav(`#/run/${payload.run_id}`);
  } catch (err) {
    errBox.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  }
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  try {
    const [state, scen] = await Promise.all([api('/api/state'), api('/api/scenarios')]);
    S.state = state;
    S.scenarios = scen.scenarios;
    $badge.innerHTML = `<span class="dot"></span>${esc(state.backend.kind)} · ${esc(state.backend.model_id)} · s1core v${esc(state.version)}`;
    $badge.classList.toggle('live', state.backend.kind === 'live');
    $footerMeta.textContent = `schema ${state.schema_version} · 127.0.0.1 only · the gate never lives in the browser`;
  } catch (err) {
    $badge.textContent = 'server unreachable';
    $app.innerHTML = `<div class="error-box">cannot reach the server: ${esc(err.message)}</div>`;
    return;
  }
  window.addEventListener('hashchange', route);
  route();
}

boot();
