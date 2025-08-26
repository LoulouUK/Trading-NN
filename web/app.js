const API_PORT = 8001; // backend uvicorn port
const API_BASE = `http://127.0.0.1:${API_PORT}`;

const API = (path, { params = {}, method = 'GET', json = null } = {}) => {
  const u = new URL(`${API_BASE}${path}`);
  Object.entries(params).forEach(([k,v]) => u.searchParams.set(k, v));
  const init = { mode: 'cors', method, headers: {} };
  if (json !== null) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  }
  return fetch(u, init);
};

let chart;

function renderChart(labelsPast, closesPast, pBuy, price, slPrice, labelsFuture, forecastY, tpPrice){
  const ctx = document.getElementById('chart').getContext('2d');
  if(chart) chart.destroy();

  const BUY_THRESHOLD = 0.5;
  const buyHit = Number(pBuy) >= BUY_THRESHOLD;

  // --- Trim past history to make forecast prominent ---
  // Keep the last N past candles, where N scales with horizon (future length)
  const futureLen = Array.isArray(labelsFuture) ? labelsFuture.length : 0;
  const desiredPast = Math.max(80, futureLen * 4); // e.g., keep ~4x horizon, but at least 80 bars
  const sliceStart = Math.max(0, labelsPast.length - desiredPast);
  labelsPast = labelsPast.slice(sliceStart);
  closesPast = closesPast.slice(sliceStart);

  // Build combined labels array (past + future)
  const labels = [...labelsPast, ...labelsFuture];

  // Past dataset (then nulls for future length)
  const pastData = [...closesPast, ...Array(labelsFuture.length).fill(null)];

  // Forecast dataset (nulls for past length, then predictions)
  const forecastData = [...Array(labelsPast.length).fill(null), ...forecastY];

  const lastPastX = labelsPast[labelsPast.length - 1];

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Close',
          data: pastData,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.25,
          borderColor: 'rgba(74,163,255,0.6)' // softer so forecast pops
        },
        {
          label: 'Forecast',
          data: forecastData,
          borderWidth: 3.5,
          pointRadius: 0,
          tension: 0.25,
          borderColor: '#8ef9a5',      // bright neon green
          borderDash: [],              // solid line for prominence
          spanGaps: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#e6e8f0' } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(2)}`
          }
        },
        annotation: {
          annotations: {
            // Vertical separator at "now"
            nowLine: {
              type: 'line',
              xMin: labelsPast.length - 0.5,
              xMax: labelsPast.length - 0.5,
              borderColor: 'rgba(255,255,255,.35)',
              borderWidth: 2,
              borderDash: [2,2]
            },
            // Buy marker at last real bar
            ...(buyHit ? {
              buyPoint: {
                type: 'point',
                xValue: lastPastX,
                yValue: Number(price),
                radius: 9,
                backgroundColor: '#21d07a',
                borderColor: '#0b6136',
                borderWidth: 2,
                pointStyle: 'triangle',
                rotation: 0
              },
              buyLabel: {
                type: 'label',
                xValue: lastPastX,
                yValue: Number(price),
                content: ['BUY'],
                position: 'top',
                backgroundColor: 'rgba(33, 208, 122, .15)',
                color: '#21d07a',
                font: { weight: '700' },
                padding: 4,
                yAdjust: -18
              }
            } : {}),
            // Stop loss line across entire chart
            slLine: {
              type: 'line',
              yMin: Number(slPrice),
              yMax: Number(slPrice),
              borderColor: '#ff6b6b',
              borderWidth: 2,
              borderDash: [6,4]
            },
            slLabel: {
              type: 'label',
              xValue: lastPastX,
              yValue: Number(slPrice),
              content: ['SL'],
              position: 'right',
              backgroundColor: 'rgba(255, 107, 107, .15)',
              color: '#ff6b6b',
              font: { weight: '700' },
              padding: 4,
              xAdjust: 20
            }
            ,
            // Take profit line across entire chart
            tpLine: {
              type: 'line',
              yMin: Number(tpPrice),
              yMax: Number(tpPrice),
              borderColor: '#8ef9a5',
              borderWidth: 2,
              borderDash: [3,3]
            },
            tpLabel: {
              type: 'label',
              xValue: lastPastX,
              yValue: Number(tpPrice),
              content: ['TP'],
              position: 'right',
              backgroundColor: 'rgba(142, 249, 165, .15)',
              color: '#8ef9a5',
              font: { weight: '700' },
              padding: 4,
              xAdjust: 20
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#9aa1b3', maxRotation: 45, minRotation: 30, autoSkip: true },
          grid: { color: 'rgba(255,255,255,.06)' }
        },
        y: {
          ticks: { color: '#9aa1b3' },
          grid: { color: 'rgba(255,255,255,.06)' }
        }
      }
    }
  });
}

// --- helpers ---
function tsToLabel(ms){
  try{ const d = new Date(ms); return d.toLocaleString(); }catch(_){ return String(ms); }
}

// Cache DOM elements once
const els = {
  symbol: document.getElementById('symbol'),
  interval: document.getElementById('interval'),
  horizon: document.getElementById('horizon'),
  btnSignal: document.getElementById('btnSignal'),
  btnTrain: document.getElementById('btnTrain'),
  status: document.getElementById('status'),
  pbuy: document.getElementById('pbuy'),
  price: document.getElementById('price'),
  atr: document.getElementById('atr'),
  sl: document.getElementById('sl'),
  tpRR: document.getElementById('tpRR'),
  tp: document.getElementById('tp'),
  tpPresets: document.getElementById('tpPresets'),
  slCard: document.querySelector('#sl').closest('.card'),
  tpCard: document.querySelector('#tp').closest('.card'),
  tableBody: document.querySelector('#signalsTable tbody')
};

async function getSignal(){
  try{
    els.status.textContent = 'Fetching signal…';
    const q = {
      symbol: els.symbol.value.trim(),
      interval: els.interval.value,
      limit: 600,
      horizon: Number(els.horizon.value) || 12,
      artifacts: 'artifacts',
      tp_rr: Number(els.tpRR?.value) || 1.5,
    };

    const res = await API('/api/signal', { params: q });
    if(!res.ok) throw new Error(`Signal HTTP ${res.status}`);
    const data = await res.json();

    // update cards
    els.pbuy.textContent = `${(Number(data.p_buy)*100).toFixed(1)}%`;
    els.price.textContent = Number(data.price).toFixed(2);
    els.atr.textContent = Number(data.atr).toFixed(2);
    els.sl.textContent = Number(data.sl_price).toFixed(2);
    els.tp.textContent = Number(data.tp_price).toFixed(2);

    // color cards based on direction
    if (data.direction === 1) {
      els.slCard.classList.remove('card-red');
      els.tpCard.classList.remove('card-green');
      els.slCard.classList.add('card-red');
      els.tpCard.classList.add('card-green');
    } else {
      els.slCard.classList.remove('card-red');
      els.tpCard.classList.remove('card-green');
      els.slCard.classList.add('card-green');
      els.tpCard.classList.add('card-red');
    }

    // chart
    const labelsPast = (data.series?.t || []).map(tsToLabel);
    const labelsFuture = (data.forecast?.t || []).map(tsToLabel);
    renderChart(labelsPast, data.series.close, data.p_buy, data.price, data.sl_price, labelsFuture, data.forecast?.y || [], data.tp_price);

    // table row
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${new Date().toLocaleString()}</td>
                    <td>${q.symbol}</td>
                    <td>${q.interval}</td>
                    <td>${(Number(data.p_buy)*100).toFixed(1)}%</td>
                    <td>${Number(data.price).toFixed(2)}</td>
                    <td>${Number(data.sl_price).toFixed(2)}</td>
                    <td>${Number(data.tp_price).toFixed(2)}</td>`;
    els.tableBody.prepend(tr);

    els.status.textContent = `Signal OK — horizon ${q.horizon} bars, features: ${data.features_used?.length ?? 'n/a'}`;
  }catch(err){
    console.error(err);
    els.status.textContent = `Error fetching signal: ${err?.message || err}`;
  }
}

async function trainModel(){
  try{
    els.status.textContent = 'Training started…';
    els.btnTrain.disabled = true; els.btnSignal.disabled = true;

    const payload = {
      symbol: els.symbol.value.trim(),
      interval: els.interval.value,
      limit: 1000,
      horizon: Number(els.horizon.value) || 12,
      epochs: 12,
      lr: 0.001,
      artifacts: 'artifacts'
    };

    const res = await API('/api/train', { method: 'POST', json: payload });
    if(!res.ok) throw new Error(`Train HTTP ${res.status}`);
    const out = await res.json();

    els.status.textContent = `Training complete — epochs: ${out.epochs}, horizon: ${out.horizon}`;
  }catch(err){
    console.error(err);
    els.status.textContent = `Error training: ${err?.message || err}`;
  }finally{
    els.btnTrain.disabled = false; els.btnSignal.disabled = false;
  }
}

// Wire buttons (after DOM is parsed)
if(document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', () => {
    els.btnSignal?.addEventListener('click', getSignal);
    els.btnTrain?.addEventListener('click', trainModel);
  });
} else {
  els.btnSignal?.addEventListener('click', getSignal);
  els.btnTrain?.addEventListener('click', trainModel);
}

// TP presets dropdown
els.tpPresets?.addEventListener('change', (e) => {
  if (e.target.value) {
    els.tpRR.value = e.target.value;
  }
});