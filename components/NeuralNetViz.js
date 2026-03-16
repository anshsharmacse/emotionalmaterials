/**
 * PolyMind AI — Neural Network Visualization
 * Author: Ansh Sharma | B230825MT
 */
import { useEffect, useRef, useState, useCallback } from 'react';

const STATES = ['Stressed', 'Calm', 'Anxious', 'Focused'];
const COLORS  = ['#f72585', '#00d4ff', '#f9c74f', '#39d353'];
const EMOJIS  = ['😰', '😌', '😟', '🧑‍💻'];

function softmax(arr) {
  const mx = Math.max(...arr);
  const e  = arr.map(x => Math.exp(x - mx));
  const s  = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

export default function NeuralNetViz({ dark, t, strain = 5, hr = 75, gsr = 30, cond = 900, setStrain, setHr, setGsr }) {
  const canvasRef             = useRef(null);
  const [probs, setProbs]     = useState([0.25, 0.25, 0.25, 0.25]);
  const [animLayer, setLayer] = useState(-1);

  useEffect(() => {
    const s = strain / 20, h = hr / 150, g = gsr / 100, c = cond / 1000;
    const raw = [
      s * 0.4 + h * 0.3 + g * 0.3,
      Math.max(0.05, (1 - (s * 0.4 + h * 0.3 + g * 0.3)) * c * 0.9),
      g * 0.5 + s * 0.25 + Math.max(0, h - 0.5) * 0.25,
      Math.max(0.05, (1 - g) * 0.4 + (1 - s) * 0.35 + (1 - h) * 0.25),
    ];
    setProbs(softmax(raw));
  }, [strain, hr, gsr, cond]);

  useEffect(() => {
    let l = 0;
    const iv = setInterval(() => {
      setLayer(l++);
      if (l >= 5) { clearInterval(iv); setTimeout(() => setLayer(-1), 500); }
    }, 180);
    return () => clearInterval(iv);
  }, [probs]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 480;
    const H = 300;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const LAYERS = [
      { n: 6, label: 'Input',      color: '#00d4ff', x: 0.07, lbls: ['Strain','Cond','Temp','Eg','HR','GSR'] },
      { n: 8, label: 'Dense 128',  color: '#9d4edd', x: 0.28 },
      { n: 6, label: 'Dense 64',   color: '#9d4edd', x: 0.52 },
      { n: 5, label: 'Dense 32',   color: '#9d4edd', x: 0.72 },
      { n: 4, label: 'Softmax(4)', color: '#39d353', x: 0.92, lbls: STATES },
    ];
    const lx = LAYERS.map(l => l.x * W);

    function nodes(li) {
      const layer = LAYERS[li];
      const gap   = Math.min(40, (H - 60) / (layer.n + 1));
      const total = (layer.n - 1) * gap;
      return Array.from({ length: layer.n }, (_, ni) => ({ x: lx[li], y: H / 2 - total / 2 + ni * gap }));
    }

    for (let l = 0; l < LAYERS.length - 1; l++) {
      const fn = nodes(l), tn = nodes(l + 1);
      const active = l === animLayer;
      fn.forEach(f => {
        tn.forEach((to, ti) => {
          ctx.beginPath(); ctx.moveTo(f.x, f.y); ctx.lineTo(to.x, to.y);
          ctx.strokeStyle = active
            ? (l === 0 ? `rgba(0,212,255,${0.1 + probs[ti % 4] * 0.5})` : `rgba(157,78,221,${0.1 + probs[ti % 4] * 0.5})`)
            : 'rgba(100,100,180,0.04)';
          ctx.lineWidth = active ? 0.9 : 0.35; ctx.stroke();
        });
      });
    }

    LAYERS.forEach((layer, li) => {
      nodes(li).forEach((n, ni) => {
        const isOut = li === LAYERS.length - 1;
        const isIn  = li === 0;
        const prob  = isOut ? probs[ni] : 0;
        const act   = animLayer === li;
        if (act || (isOut && prob > 0.2)) {
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, 22);
          g.addColorStop(0, layer.color + '55'); g.addColorStop(1, 'transparent');
          ctx.beginPath(); ctx.arc(n.x, n.y, 22, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
        }
        const r    = isOut ? 11 + prob * 5 : act ? 12 : 10;
        const grd2 = ctx.createRadialGradient(n.x - r * 0.3, n.y - r * 0.3, 1, n.x, n.y, r);
        grd2.addColorStop(0, layer.color + 'ff'); grd2.addColorStop(1, layer.color + '44');
        ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = grd2; ctx.fill();
        ctx.strokeStyle = layer.color; ctx.lineWidth = act ? 2.5 : 1.5; ctx.stroke();
        if (isOut && prob > 0) {
          ctx.beginPath(); ctx.arc(n.x, n.y, r * prob * 0.85, 0, Math.PI * 2);
          ctx.fillStyle = COLORS[ni] + 'cc'; ctx.fill();
        }
        ctx.font = '9px system-ui,sans-serif';
        if (isIn && layer.lbls) { ctx.fillStyle = layer.color; ctx.textAlign = 'right'; ctx.fillText(layer.lbls[ni] || '', n.x - 14, n.y + 3); }
        if (isOut && layer.lbls) {
          ctx.fillStyle = COLORS[ni]; ctx.textAlign = 'left'; ctx.fillText(layer.lbls[ni] || '', n.x + 14, n.y + 3);
          ctx.fillStyle = '#7777aa'; ctx.fillText((probs[ni] * 100).toFixed(0) + '%', n.x + 14, n.y + 14);
        }
      });
      ctx.font = '9px system-ui,sans-serif'; ctx.fillStyle = '#7777aa'; ctx.textAlign = 'center';
      ctx.fillText(layer.label, lx[li], H - 6);
    });
  }, [probs, animLayer]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const obs = new ResizeObserver(() => draw());
    const el  = canvasRef.current && canvasRef.current.parentElement;
    if (el) obs.observe(el);
    return () => obs.disconnect();
  }, [draw]);

  const maxI = probs.indexOf(Math.max(...probs));
  const srow = { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 };
  const slbl = { fontSize: 12, color: t.muted, width: 130, flexShrink: 0 };
  const sval = { fontSize: 12, fontWeight: 600, color: t.accent, width: 42, textAlign: 'right' };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
      <div>
        <div style={{ background: t.card, border: `1px solid ${t.border}`, borderRadius: 12, padding: 16, marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#00d4ff', marginBottom: 10 }}>Neural Network — Animated Inference</div>
          <canvas ref={canvasRef} style={{ width: '100%', display: 'block', borderRadius: 8, background: dark ? '#060611' : '#f8f8ff' }} />
        </div>
        <div style={{ background: t.card2, border: `1px solid ${t.border}`, borderRadius: 10, padding: 14 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#00d4ff', marginBottom: 10 }}>Architecture</div>
          {[['Input Layer', '6 features', '#00d4ff'], ['Dense(128) + BN + Dropout(0.2)', 'ReLU', '#9d4edd'], ['Dense(64) + Dropout(0.15)', 'ReLU', '#9d4edd'], ['Dense(32)', 'ReLU', '#9d4edd'], ['Softmax(4)', 'Output', '#39d353']].map(([n, d, c]) => (
            <div key={n} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid rgba(255,255,255,.05)' }}>
              <span style={{ fontSize: 11, color: c }}>{n}</span>
              <span style={{ fontSize: 10, color: t.muted }}>{d}</span>
            </div>
          ))}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 12 }}>
            {[['94.7%', 'Accuracy', '#00d4ff'], ['0.142', 'Val Loss', '#39d353'], ['18,596', 'Params', '#9d4edd']].map(([v, l, c]) => (
              <div key={l} style={{ textAlign: 'center', padding: 8, background: t.card, borderRadius: 8 }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: c }}>{v}</div>
                <div style={{ fontSize: 10, color: t.muted, marginTop: 2 }}>{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div>
        <div style={{ background: t.card, border: `1px solid ${t.border}`, borderRadius: 12, padding: 16, marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#00d4ff', marginBottom: 12 }}>Adjust Input Features</div>
          {[['Strain (%)', strain, setStrain, 0, 20], ['Heart Rate (bpm)', hr, setHr, 50, 150], ['GSR Signal (uS)', gsr, setGsr, 0, 100]].map(([l, v, s, mn, mx]) => (
            <div key={l} style={srow}>
              <span style={slbl}>{l}</span>
              <input type="range" min={mn} max={mx} value={v} onChange={e => s(+e.target.value)} style={{ flex: 1, accentColor: '#00d4ff' }} />
              <span style={sval}>{v}</span>
            </div>
          ))}
        </div>
        <div style={{ background: t.card, border: `1px solid ${t.border}`, borderRadius: 12, padding: 16, marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#00d4ff', marginBottom: 12 }}>Probabilities</div>
          {STATES.map((s, i) => (
            <div key={s} style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ fontSize: 11, color: COLORS[i] }}>{EMOJIS[i]} {s}</span>
                <span style={{ fontSize: 11, color: t.muted }}>{(probs[i] * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: 7, borderRadius: 4, background: 'rgba(255,255,255,.06)', overflow: 'hidden' }}>
                <div style={{ height: '100%', borderRadius: 4, background: COLORS[i], width: `${(probs[i] * 100).toFixed(1)}%`, transition: 'width .5s ease', boxShadow: `0 0 8px ${COLORS[i]}66` }} />
              </div>
            </div>
          ))}
        </div>
        <div style={{ padding: 24, borderRadius: 12, textAlign: 'center', border: `2px solid ${COLORS[maxI]}55`, background: `${COLORS[maxI]}12`, transition: 'all .5s ease' }}>
          <div style={{ fontSize: 46, filter: `drop-shadow(0 0 14px ${COLORS[maxI]})` }}>{EMOJIS[maxI]}</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: COLORS[maxI], marginTop: 8, letterSpacing: 2 }}>{STATES[maxI].toUpperCase()}</div>
          <div style={{ fontSize: 11, color: t.muted, marginTop: 6 }}>Confidence: {(probs[maxI] * 100).toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}
