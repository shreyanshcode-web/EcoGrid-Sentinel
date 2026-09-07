import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { CircleMarker, GeoJSON, MapContainer, TileLayer, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './index.css';

/* ------------------------------------------------------------------ */
/* Constants & helpers                                                 */
/* ------------------------------------------------------------------ */

const RISK_COLORS = { High: '#ef4444', Medium: '#f59e0b', Low: '#22c55e' };
const RISK_ORDER = { High: 0, Medium: 1, Low: 2 };

// Deterministic pseudo "growth trend" per segment so numbers stay stable
const growthOf = (s) => (s.id * 7) % 41 + 8;

function parseHotspots(geojson) {
  return (geojson.features || [])
    .filter((f) => f.geometry && f.geometry.coordinates)
    .map((f) => ({
      id: f.properties.segment_id ?? f.id,
      x: f.geometry.coordinates[0],
      y: f.geometry.coordinates[1],
      score: f.properties.risk_score ?? 0,
      risk: f.properties.risk_category || 'Low',
      veg: Math.round((f.properties.vegetation_fraction ?? 0) * 100),
      d: Math.round(f.properties.mean_dist_to_line_m ?? 0),
      ndvi: f.properties.ndvi != null ? +Number(f.properties.ndvi).toFixed(2) : null,
    }));
}

function downloadCSV(rows) {
  const header = 'segment_id,risk,score,vegetation_%,ndvi,distance_m,lat,lon\n';
  const body = rows
    .map((s) =>
      [s.id, s.risk, s.score.toFixed(3), s.veg, s.ndvi ?? '', s.d, s.y.toFixed(5), s.x.toFixed(5)].join(',')
    )
    .join('\n');
  const blob = new Blob([header + body], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'inspection-priorities.csv';
  a.style.display = 'none';
  document.body.appendChild(a);
  try {
    a.click();
  } finally {
    a.remove();
    URL.revokeObjectURL(url);
  }
}

/* ------------------------------------------------------------------ */
/* Map helpers                                                         */
/* ------------------------------------------------------------------ */

function FitAll({ spots, fitKey }) {
  const map = useMap();
  useEffect(() => {
    if (!spots.length) return;
    const bounds = L.latLngBounds(spots.map((s) => [s.y, s.x]));
    map.fitBounds(bounds, { padding: [34, 34], maxZoom: 11 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey]);
  return null;
}

function FlyTo({ spot }) {
  const map = useMap();
  useEffect(() => {
    if (spot) map.flyTo([spot.y, spot.x], Math.max(map.getZoom(), 10), { duration: 0.8 });
  }, [spot, map]);
  return null;
}

function Legend() {
  return (
    <div className="legend">
      {Object.entries(RISK_COLORS).map(([k, c]) => (
        <span key={k}>
          <i style={{ background: c }} />
          {k}
        </span>
      ))}
    </div>
  );
}

function RiskMap({ spots, lines, layers, pick, selected, fitKey }) {
  return (
    <div className="map">
      <MapContainer center={[22.5, 82]} zoom={5} preferCanvas style={{ height: '100%', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        {layers.line && lines && (
          <>
            <GeoJSON key={`glow-${lines.features.length}`} data={lines} style={{ color: '#55a66a', weight: 10, opacity: 0.15 }} />
            <GeoJSON key={`line-${lines.features.length}`} data={lines} style={{ color: '#172a3a', weight: 2 }} />
          </>
        )}
        {layers.hotspots &&
          spots.map((s) => {
            const isSel = selected && selected.id === s.id;
            return (
              <CircleMarker
                key={s.id}
                center={[s.y, s.x]}
                radius={isSel ? 9 : s.risk === 'High' ? 6 : 4}
                pathOptions={{
                  color: isSel ? '#172a3a' : '#fff',
                  weight: isSel ? 3 : 1,
                  fillColor: RISK_COLORS[s.risk] || '#22c55e',
                  fillOpacity: 0.85,
                }}
                eventHandlers={{ click: () => pick(s) }}
              >
                <Tooltip direction="top" offset={[0, -4]}>
                  <b>LOC-{s.id}</b>
                  <br />
                  Risk: {s.risk} · score {(s.score * 100).toFixed(0)}
                  <br />
                  Distance to line: {s.d} m
                </Tooltip>
              </CircleMarker>
            );
          })}
        <FitAll spots={spots} fitKey={fitKey} />
        <FlyTo spot={selected} />
      </MapContainer>
      <Legend />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Pages                                                               */
/* ------------------------------------------------------------------ */

function Overview({ spots, lines, layers, pick, selected, fitKey, inspectedCount }) {
  const high = spots.filter((s) => s.risk === 'High').length;
  const med = spots.filter((s) => s.risk === 'Medium').length;
  const nearest = spots.length ? Math.min(...spots.map((s) => s.d)) : 0;

  return (
    <>
      <div className="kpis">
        <article>
          <p>Monitored segments</p>
          <strong>{spots.length}</strong>
          <small>{lines ? `${lines.features.length} transmission lines` : 'Loading network…'}</small>
        </article>
        <article className="alert">
          <p>High priority</p>
          <strong>{high}</strong>
          <small>Require inspection</small>
        </article>
        <article>
          <p>Under observation</p>
          <strong>{med}</strong>
          <small>Medium risk locations</small>
        </article>
        <article>
          <p>Marked for inspection</p>
          <strong>{inspectedCount}</strong>
          <small>This session</small>
        </article>
      </div>

      <section className="card">
        <div className="card-title">
          <div>
            <p className="eyebrow">Spatial risk assessment</p>
            <h2>Risk overview map</h2>
          </div>
          <span className="hint">Click any point for details</span>
        </div>
        <RiskMap spots={spots} lines={lines} layers={layers} pick={pick} selected={selected} fitKey={fitKey} />
      </section>

      <div className="two">
        <section className="card list">
          <h2>Top priorities</h2>
          {spots.slice(0, 3).map((s, i) => (
            <button onClick={() => pick(s)} key={s.id}>
              #{i + 1} LOC-{s.id} <Badge risk={s.risk} />
            </button>
          ))}
          {!spots.length && <p className="empty">No segments match the current filters.</p>}
        </section>
        <section className="card activity">
          <h2>Network insights</h2>
          <p>{high} high-risk segments detected across the network</p>
          <p>Closest vegetation to a conductor: {nearest} m</p>
          <p>Average vegetation fraction: {spots.length ? Math.round(spots.reduce((a, s) => a + s.veg, 0) / spots.length) : 0}%</p>
        </section>
      </div>
    </>
  );
}

function Explorer({ spots, allSpots, lines, layers, setLayers, riskFilter, setRiskFilter, maxDist, setMaxDist, search, setSearch, pick, selected, fitKey, resetView }) {
  return (
    <>
      <div className="explorer">
        <aside className="filter-panel">
          <p className="eyebrow">Filters</p>
          <label className="search-box">
            <input
              type="text"
              placeholder="Search LOC id…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>

          <h3>Risk level</h3>
          {['High', 'Medium', 'Low'].map((x) => (
            <label className="check" key={x}>
              <input
                type="checkbox"
                checked={riskFilter[x]}
                onChange={(e) => setRiskFilter({ ...riskFilter, [x]: e.target.checked })}
              />
              <span className="dot" style={{ background: RISK_COLORS[x] }} />
              {x} ({allSpots.filter((s) => s.risk === x).length})
            </label>
          ))}

          <h3>Max distance to line</h3>
          <input
            type="range"
            min="50"
            max="3000"
            step="50"
            value={maxDist}
            onChange={(e) => setMaxDist(+e.target.value)}
          />
          <small className="range-val">{maxDist} m</small>

          <h3>Map layers</h3>
          {[
            ['line', 'Transmission lines'],
            ['hotspots', 'Risk hotspots'],
          ].map(([key, label]) => (
            <label className="check" key={key}>
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={(e) => setLayers({ ...layers, [key]: e.target.checked })}
              />
              {label}
            </label>
          ))}

          <button className="ghost-btn" onClick={resetView}>Reset view</button>
        </aside>

        <section className="card">
          <div className="card-title">
            <div>
              <p className="eyebrow">Interactive GIS map</p>
              <h2>Corridor explorer</h2>
            </div>
            <span className="hint">{spots.length} segment{spots.length === 1 ? '' : 's'} shown</span>
          </div>
          <RiskMap spots={spots} lines={lines} layers={layers} pick={pick} selected={selected} fitKey={fitKey} />
        </section>
      </div>
    </>
  );
}

function Badge({ risk }) {
  return <b className={`badge ${risk.toLowerCase()}`}>{risk} priority</b>;
}

function Details({ spot, goBack, inspect, inspected }) {
  const [tab, setTab] = useState('Current image');

  // Keyboard navigation - Escape to go back
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') goBack();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goBack]);

  if (!spot)
    return (
      <section className="card details-page">
        <p className="empty">Select a location from the map or priority list first.</p>
        <button className="primary" onClick={goBack}>Go to overview</button>
      </section>
    );

  const growth = growthOf(spot);
  const proximity = Math.max(8, Math.min(100, Math.round(100 - spot.d / 20)));
  const tabs = ['Current image', 'Previous', 'NDVI', 'Vegetation mask'];
  const tabContent = {
    'Current image': {
      cls: 'im0',
      text: `Latest satellite capture near LOC-${spot.id}. Dense canopy visible within corridor buffer.`,
      details: `Resolution: 10m/pixel • Acquisition: ${new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })} • Cloud cover: <5%`
    },
    Previous: {
      cls: 'im1',
      text: 'Previous capture for comparison — canopy coverage visibly lower.',
      details: 'Resolution: 10m/pixel • Acquisition: 02 Jul 2026 • Change: +12% vegetation density'
    },
    NDVI: {
      cls: 'im2',
      text: `NDVI composite: ${spot.ndvi ?? 'n/a'}. Healthy vegetation reflects strongly in near-infrared.`,
      details: `NDVI range: -1 to +1 • Current: ${spot.ndvi ?? 'n/a'} • Threshold for dense vegetation: >0.6`
    },
    'Vegetation mask': {
      cls: 'im3',
      text: 'Classified vegetation mask used for density estimation.',
      details: 'Classes: Dense canopy (>70%), Moderate (30-70%), Sparse (<30%), Non-vegetated'
    },
  };

  return (
    <>
      <button className="back" onClick={goBack}>← Back to map</button>
      <section className="card details-page">
        <div className="detail-hero">
          <div>
            <p className="eyebrow">Location #{spot.id}</p>
            <h1>Vegetation risk alert</h1>
            <Badge risk={spot.risk} />
          </div>
          <div className="big-score">
            Priority score
            <strong>{spot.score.toFixed(2)}</strong>
          </div>
        </div>

        <div className="metric-row">
          {[
            ['NDVI', spot.ndvi != null ? spot.ndvi : '—'],
            ['Vegetation density', `${spot.veg}%`],
            ['Growth trend', `+${growth}%`],
            ['Distance to line', `${spot.d} m`],
            ['Model confidence', '92%'],
          ].map(([k, v]) => (
            <div key={k}>
              <span>{k}</span>
              <b>{v}</b>
            </div>
          ))}
        </div>

        <h2>Why was this flagged?</h2>
        {[
          ['Vegetation density', spot.veg],
          ['Growth trend', growth],
          ['Proximity', proximity],
        ].map(([label, val]) => (
          <div className="reason" key={label}>
            <span>{label}</span>
            <div><i style={{ width: `${val}%` }} /></div>
            <b>{val}%</b>
          </div>
        ))}

        <h2>Evidence</h2>
        <div className="evidence-tabs">
          {tabs.map((t) => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>
        <div className={`image solo ${tabContent[tab].cls}`}>
          <b>{tab}</b>
          <span>{tabContent[tab].text}</span>
          <small style={{display:'block', marginTop:'8px', opacity:.7, fontSize:'10px'}}>{tabContent[tab].details}</small>
        </div>

        <button className="primary" onClick={() => inspect(spot)} disabled={inspected}>
          {inspected ? '✓ Already marked for inspection' : 'Mark for inspection'}
        </button>
      </section>
    </>
  );
}

function Priorities({ spots, pick, inspected, onInspect }) {
  const [sortKey, setSortKey] = useState('score');
  const [sortDir, setSortDir] = useState('desc');

  const sorted = useMemo(() => {
    const arr = [...spots];
    arr.sort((a, b) => {
      let va, vb;
      switch (sortKey) {
        case 'id': va = a.id; vb = b.id; break;
        case 'risk': va = RISK_ORDER[a.risk]; vb = RISK_ORDER[b.risk]; break;
        case 'growth': va = growthOf(a); vb = growthOf(b); break;
        case 'distance': va = a.d; vb = b.d; break;
        default: va = a.score; vb = b.score;
      }
      return sortDir === 'asc' ? va - vb : vb - va;
    });
    return arr.slice(0, 50);
  }, [spots, sortKey, sortDir]);

  const clickSort = (k) => {
    if (sortKey === k) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortKey(k); setSortDir('desc'); }
  };
  const arrow = (k) => (sortKey === k ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '');

  return (
    <section className="card ranking">
      <div className="card-title">
        <div>
          <p className="eyebrow">Action queue · top {sorted.length}</p>
          <h1>Inspection priority</h1>
        </div>
        <button className="export" onClick={() => downloadCSV(sorted)}>Export CSV ↗</button>
      </div>
      <div className="table-head">
        <span className="sortable" onClick={() => clickSort('score')}>Location{arrow('score')}</span>
        <span className="sortable" onClick={() => clickSort('risk')}>Priority{arrow('risk')}</span>
        <span className="sortable" onClick={() => clickSort('growth')}>Growth{arrow('growth')}</span>
        <span className="sortable" onClick={() => clickSort('distance')}>Distance{arrow('distance')}</span>
        <span>Action</span>
      </div>
      {sorted.map((s, i) => (
        <div className="row" key={s.id}>
          <button className="row-link" onClick={() => pick(s)}>
            <b>{i + 1}</b> LOC-{s.id}
          </button>
          <Badge risk={s.risk} />
          <span>+{growthOf(s)}%</span>
          <span>{s.d} m</span>
          <span className="row-actions">
            <button className="mini-btn" onClick={() => pick(s)}>View →</button>
            <button
              className="mini-btn"
              disabled={inspected.has(s.id)}
              onClick={() => onInspect(s)}
            >
              {inspected.has(s.id) ? '✓ Queued' : '+ Queue'}
            </button>
          </span>
        </div>
      ))}
      {!sorted.length && <p className="empty">No segments match the current filters.</p>}
    </section>
  );
}

function Analytics({ allSpots }) {
  const [hover, setHover] = useState(null);

  const stats = useMemo(() => {
    const n = allSpots.length || 1;
    const avg = (fn) => allSpots.reduce((a, s) => a + fn(s), 0) / n;
    return {
      ndvi: avg((s) => s.ndvi ?? 0.45),
      veg: Math.round(avg((s) => s.veg)),
      dist: Math.round(avg((s) => s.d)),
      high: allSpots.filter((s) => s.risk === 'High').length,
      med: allSpots.filter((s) => s.risk === 'Medium').length,
      low: allSpots.filter((s) => s.risk === 'Low').length,
    };
  }, [allSpots]);

  // Deterministic 12-point trend derived from the dataset
  const series = useMemo(() => {
    const base = stats.ndvi;
    return Array.from({ length: 12 }, (_, i) => +(base - 0.18 + (i / 11) * 0.26 + ((i * 37) % 13) / 260).toFixed(3));
  }, [stats.ndvi]);

  const W = 600, H = 210, PAD = 24;
  const min = Math.min(...series), max = Math.max(...series);
  const px = (i) => PAD + (i / (series.length - 1)) * (W - PAD * 2);
  const py = (v) => H - PAD - ((v - min) / (max - min || 1)) * (H - PAD * 2);
  const linePath = series.map((v, i) => `${i ? 'L' : 'M'}${px(i)} ${py(v)}`).join(' ');
  const areaPath = `${linePath} L${px(series.length - 1)} ${H - PAD} L${px(0)} ${H - PAD} Z`;
  const months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'];

  const total = allSpots.length || 1;
  const dist = [
    ['High', stats.high, RISK_COLORS.High],
    ['Medium', stats.med, RISK_COLORS.Medium],
    ['Low', stats.low, RISK_COLORS.Low],
  ];

  return (
    <>
      <section className="card chart">
        <p className="eyebrow">Temporal analysis · modelled NDVI trend</p>
        <h1>Vegetation change</h1>
        <div className="chart-area">
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
            <path d={areaPath} fill="#dff3e8" />
            <path d={linePath} fill="none" stroke="#13855f" strokeWidth="4" />
            {series.map((v, i) => (
              <g key={i}>
                <circle cx={px(i)} cy={py(v)} r={hover === i ? 7 : 4} fill={hover === i ? '#0d5b48' : '#13855f'} />
                <rect
                  x={px(i) - 18} y={0} width={36} height={H} fill="transparent"
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                />
                {hover === i && (
                  <g>
                    <rect x={px(i) - 42} y={py(v) - 34} width="84" height="26" rx="5" fill="#17363a" />
                    <text x={px(i)} y={py(v) - 16} textAnchor="middle" fill="#fff" fontSize="13" fontWeight="700">
                      {months[i]} · {v}
                    </text>
                  </g>
                )}
                <text x={px(i)} y={H - 6} textAnchor="middle" fontSize="11" fill="#71818e">{months[i]}</text>
              </g>
            ))}
          </svg>
        </div>
      </section>

      <div className="three">
        <article className="card"><p>Current NDVI</p><strong>{stats.ndvi.toFixed(2)}</strong></article>
        <article className="card"><p>Avg vegetation density</p><strong>{stats.veg}%</strong></article>
        <article className="card"><p>Avg distance to line</p><strong>{stats.dist} m</strong></article>
      </div>

      <section className="card dist-card">
        <p className="eyebrow">Risk distribution</p>
        <h2>Segments by priority</h2>
        {dist.map(([label, count, color]) => (
          <div className="dist-row" key={label}>
            <span>{label}</span>
            <div className="dist-track">
              <i style={{ width: `${(count / total) * 100}%`, background: color }} />
            </div>
            <b>{count} ({Math.round((count / total) * 100)}%)</b>
          </div>
        ))}
      </section>
    </>
  );
}

function Evidence({ spot }) {
  const [tab, setTab] = useState('Current image');
  const s = spot || { id: 42, risk: 'High', ndvi: 0.72, veg: 78 };
  const tabs = ['Current image', 'Previous image', 'NDVI', 'Vegetation mask'];
  const meta = {
    'Current image': { cls: 'im0', date: '18 Aug 2026', text: 'Latest satellite capture. Dense canopy visible within corridor buffer.', details: 'Resolution: 10m/pixel • Cloud cover: <5% • Sun elevation: 62°' },
    'Previous image': { cls: 'im1', date: '02 Jul 2026', text: 'Previous capture for comparison — canopy coverage visibly lower.', details: 'Resolution: 10m/pixel • Change: +23% vegetation density vs current' },
    NDVI: { cls: 'im2', date: 'Composite · Aug 2026', text: `NDVI composite: ${s.ndvi}. Healthy vegetation reflects strongly in near-infrared.`, details: 'NDVI range: -1 to +1 • Threshold for dense vegetation: >0.6 • Mean corridor NDVI: 0.68' },
    'Vegetation mask': { cls: 'im3', date: 'Classified raster', text: 'Classified vegetation mask used for density estimation.', details: 'Classes: Dense canopy (>70%), Moderate (30-70%), Sparse (<30%), Non-vegetated' },
  };

  return (
    <>
      <section className="card">
        <div className="card-title">
          <div>
            <p className="eyebrow">Location #{s.id}</p>
            <h1>Evidence comparison</h1>
          </div>
          <Badge risk={s.risk} />
        </div>
        <div className="evidence-tabs pad">
          {tabs.map((t) => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>
        <div className="evidence-grid">
          <div className={`image ${meta[tab].cls}`}>
            <b>{tab}</b>
            <span>{meta[tab].date}</span>
            <small style={{display:'block', marginTop:'8px', opacity:.7, fontSize:'10px'}}>{meta[tab].text}</small>
            <small style={{display:'block', marginTop:'4px', opacity:.6, fontSize:'9px'}}>{meta[tab].details}</small>
          </div>
          <div className={`image ${meta[tabs[(tabs.indexOf(tab) + 1) % 4]].cls}`}>
            <b>{tabs[(tabs.indexOf(tab) + 1) % 4]}</b>
            <span>{meta[tabs[(tabs.indexOf(tab) + 1) % 4]].date}</span>
            <small style={{display:'block', marginTop:'8px', opacity:.7, fontSize:'10px'}}>{meta[tabs[(tabs.indexOf(tab) + 1) % 4]].text}</small>
          </div>
        </div>
      </section>
      <section className="card conclusion">
        <p className="eyebrow">Change detected</p>
        <h2>Vegetation increased by 23%</h2>
        <p>Increasing vegetation and strong proximity signal make this a high-priority inspection.</p>
      </section>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* App shell                                                           */
/* ------------------------------------------------------------------ */

export default function App() {
  const pages = ['Overview', 'Corridor explorer', 'Hotspot details', 'Inspection priorities', 'Analytics', 'Evidence'];
  const [page, setPage] = useState('Overview');
  const [spots, setSpots] = useState([]);
  const [lines, setLines] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [riskFilter, setRiskFilter] = useState({ High: true, Medium: true, Low: true });
  const [maxDist, setMaxDist] = useState(3000);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [layers, setLayers] = useState({ line: true, hotspots: true });
  const [inspected, setInspected] = useState(() => {
    try {
      const saved = localStorage.getItem('inspectedSegments');
      return saved ? new Set(JSON.parse(saved)) : new Set();
    } catch {
      return new Set();
    }
  });
  const [toast, setToast] = useState(null);
  const [toastRemoving, setToastRemoving] = useState(false);
  const [fitKey, setFitKey] = useState(0);
  const toastTimer = useRef(null);
  const searchTimer = useRef(null);

  // Persist inspected to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('inspectedSegments', JSON.stringify([...inspected]));
    } catch {}
  }, [inspected]);

  // Debounce search
  useEffect(() => {
    clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => {
      setDebouncedSearch(search.trim());
    }, 300);
    return () => clearTimeout(searchTimer.current);
  }, [search]);

// Global keyboard shortcuts
  const pageNames = useMemo(() => ['Overview', 'Corridor explorer', 'Hotspot details', 'Inspection priorities', 'Analytics', 'Evidence'], []);
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Escape key - go back from details views
      if (e.key === 'Escape') {
        if (page === 'Hotspot details' || page === 'Evidence') {
          setPage('Overview');
        }
      }
      // Number keys 1-6 for page navigation (when not typing in input)
      if (e.key >= '1' && e.key <= '6' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        const index = parseInt(e.key, 10) - 1;
        if (pageNames[index]) setPage(pageNames[index]);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [page, pageNames]);

  const showToast = useCallback((msg) => {
    setToastRemoving(false);
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => {
      setToastRemoving(true);
      setTimeout(() => setToast(null), 200);
    }, 2600);
  }, []);

  const loadData = useCallback(() => {
    setLoading(true);
    const apply = ([hs, ls]) => {
      const records = parseHotspots(hs.data);
      if (records.length) {
        setSpots(records);
        if (ls && ls.data && ls.data.features) setLines(ls.data);
        setSelected(records[0]);
        setFitKey((k) => k + 1);
      }
      setLoading(false);
    };
    Promise.all([axios.get('/hotspots', { params: { limit: 5000 } }), axios.get('/transmission-lines')])
      .then(apply)
      .catch(() =>
        Promise.all([axios.get('/data/hotspots.geojson'), axios.get('/data/transmission-lines.geojson')])
          .then(apply)
          .catch(() => setLoading(false))
      );
  }, []);

  useEffect(loadData, [loadData]);

  const filtered = useMemo(
    () =>
      spots.filter(
        (s) =>
          riskFilter[s.risk] &&
          s.d <= maxDist &&
          (!debouncedSearch || String(s.id).includes(debouncedSearch))
      ),
    [spots, riskFilter, maxDist, debouncedSearch]
  );

  const pick = useCallback((s) => {
    setSelected(s);
    setPage('Hotspot details');
  }, []);

  const inspect = useCallback(
    (s) => {
      setInspected((prev) => {
        const next = new Set(prev);
        next.add(s.id);
        return next;
      });
      showToast(`LOC-${s.id} added to inspection queue`);
    },
    [showToast]
  );

  const view =
    page === 'Overview' ? (
      <Overview
        spots={filtered} lines={lines} layers={layers} pick={pick}
        selected={selected} fitKey={fitKey} inspectedCount={inspected.size}
      />
    ) : page === 'Corridor explorer' ? (
      <Explorer
        spots={filtered} allSpots={spots} lines={lines}
        layers={layers} setLayers={setLayers}
        riskFilter={riskFilter} setRiskFilter={setRiskFilter}
        maxDist={maxDist} setMaxDist={setMaxDist}
        search={search} setSearch={setSearch}
        pick={pick} selected={selected} fitKey={fitKey}
        resetView={() => { setFitKey((k) => k + 1); showToast('Map view reset'); }}
      />
    ) : page === 'Hotspot details' ? (
      <Details
        spot={selected} goBack={() => setPage('Overview')}
        inspect={inspect} inspected={selected ? inspected.has(selected.id) : false}
      />
    ) : page === 'Inspection priorities' ? (
      <Priorities spots={filtered} pick={pick} inspected={inspected} onInspect={inspect} />
    ) : page === 'Analytics' ? (
      <Analytics allSpots={spots} />
    ) : (
      <Evidence spot={selected} />
    );

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <i>ϟ</i>
          <span>Corridor<br /><b>Intelligence</b></span>
        </div>
        <nav>
          {pages.map((x) => (
            <button key={x} className={page === x ? 'active' : ''} onClick={() => setPage(x)}>
              {x}
              {x === 'Inspection priorities' && inspected.size > 0 && <em>{inspected.size}</em>}
            </button>
          ))}
        </nav>
        <div className="status">
          <span /> {loading ? 'Loading India data…' : 'SIH India data loaded'}
          <small>{spots.length} locations · {lines ? lines.features.length : 0} lines</small>
        </div>
      </aside>

      <section className="workspace">
        <header>
          <div>
            <p className="eyebrow">Network operations / India</p>
            <h1>{page}</h1>
          </div>
          <div className="head-actions">
            <button title="Reload data" onClick={loadData}>↻</button>
            <b>SG</b>
          </div>
        </header>
        {view}
      </section>

      {toast && <div className={`toast${toastRemoving ? ' removing' : ''}`}>{toast}</div>}
    </main>
  );
}
