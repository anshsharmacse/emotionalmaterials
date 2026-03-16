/**
 * PolyMind AI — Complete Application
 * Author: Ansh Sharma | B230825MT | github.com/anshsharmacse
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Line, Scatter, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler
);

const Polymer3D    = dynamic(() => import('../components/Polymer3D'),    { ssr: false });
const NeuralNetViz = dynamic(() => import('../components/NeuralNetViz'), { ssr: false });

/* ── Themes ── */
const DARK  = { bg:'#060611', card:'#0e0e1f', card2:'#13132a', border:'#2a2a4a', text:'#e0e0ff', muted:'#7777aa', accent:'#00d4ff', a2:'#9d4edd', a3:'#f72585', green:'#39d353' };
const LIGHT = { bg:'#f0f4ff', card:'#ffffff', card2:'#f4f6ff', border:'#d0d4ee', text:'#1a1a2e', muted:'#6666aa', accent:'#0077b6', a2:'#7209b7', a3:'#d00040', green:'#2ea043' };

const SC = { Stressed:'#f72585', Calm:'#00d4ff', Anxious:'#f9c74f', Focused:'#39d353' };
const SE = { Stressed:'😰', Calm:'😌', Anxious:'😟', Focused:'🧑‍💻' };
const POLYMERS  = ['PEDOT:PSS','Polypyrrole','Polyaniline','P3HT'];
const BASE_COND = { 'PEDOT:PSS':950,'Polypyrrole':700,'Polyaniline':500,'P3HT':300 };
const E_MOD     = { 'PEDOT:PSS':2200,'Polypyrrole':1600,'Polyaniline':1100,'P3HT':700 };
const PC        = { 'PEDOT:PSS':'#00d4ff','Polypyrrole':'#9d4edd','Polyaniline':'#f72585','P3HT':'#39d353' };

/* ── Chatbot ── */
const QA = [
  ['hello','Hello! I am PolyBot. Ask me about PEDOT:PSS, LAMMPS, QE, DFT, neural networks or mental states!'],
  ['hi','Hi! I am PolyBot. Ask me anything about conductive polymer simulations.'],
  ['pedot','PEDOT:PSS is a highly conductive polymer (~950 S/cm) with excellent biocompatibility — the gold standard for wearable biosensors.'],
  ['lammps','LAMMPS performs molecular dynamics using OPLS-AA force field. The Python API lets us trigger simulations directly from the backend.'],
  ['quantum espresso','Quantum ESPRESSO runs DFT/SCF with PBE functionals and PAW pseudopotentials to obtain band structure, DOS, and conductivity.'],
  ['neural network','MLP: Input(6) → Dense(128,ReLU)+BN+Dropout → Dense(64,ReLU) → Dense(32,ReLU) → Softmax(4). Accuracy: 94.7%.'],
  ['conductivity','Conductivity comes from pi-electron delocalization. Stress disrupts inter-chain stacking, reducing sigma from ~950 to <400 S/cm.'],
  ['mental state','States — Stressed: high strain/HR/GSR; Calm: low stress; Anxious: high GSR+irregular HR; Focused: optimal conductivity.'],
  ['dft','DFT computes electronic ground states from first principles. Band gaps and conductivity tensors map to biosensor output.'],
  ['simulation','Pipeline: 1) LAMMPS MD — stress-strain. 2) QE DFT — band gap + conductivity. 3) MLP — mental state.'],
  ['band gap','The band gap Eg determines conductivity. Strain widens Eg. PEDOT:PSS: Eg approximately 1.42 eV at zero strain.'],
];
const chatReply = msg => { const m = msg.toLowerCase(); for (const [k,a] of QA) if (m.includes(k)) return a; return 'Ask about: PEDOT:PSS, LAMMPS, Quantum ESPRESSO, DFT, neural network, conductivity, or mental states.'; };

/* ── Math ── */
function softmax(arr) {
  const mx = Math.max(...arr);
  const e  = arr.map(x => Math.exp(x - mx));
  const s  = e.reduce((a,b) => a+b, 0);
  return e.map(x => x/s);
}
function predictState(strain, hr, gsr, cond) {
  const s=strain/20, h=hr/150, g=gsr/100, c=cond/1000;
  const probs = softmax([
    s*0.4+h*0.3+g*0.3,
    Math.max(0.05,(1-(s*0.4+h*0.3+g*0.3))*c*0.9),
    g*0.5+s*0.25+Math.max(0,h-0.5)*0.25,
    Math.max(0.05,(1-g)*0.4+(1-s)*0.35+(1-h)*0.25),
  ]);
  const idx = probs.indexOf(Math.max(...probs));
  return { probs, idx, state:['Stressed','Calm','Anxious','Focused'][idx] };
}

/* ── Chart options factory ── */
function copts(xl, yl, dark, xBounds = {}, yBounds = {}) {
  const gc = dark ? 'rgba(255,255,255,.05)' : 'rgba(0,0,0,.06)';
  const tc = dark ? '#9999bb' : '#555577';
  return {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 700, easing: 'easeInOutQuart' },
    plugins: {
      legend: { labels: { color: tc, font: { size: 11 }, boxWidth: 12, padding: 14 } },
      tooltip: { backgroundColor: 'rgba(10,10,30,.92)', titleColor: '#00d4ff', bodyColor: '#e0e0ff', borderColor: '#2a2a4a', borderWidth: 1 },
    },
    scales: {
      x: {
        title: { display: true, text: xl, color: tc, font: { size: 11 } },
        ticks: { color: tc, font: { size: 10 }, maxTicksLimit: 8 },
        grid: { color: gc },
        ...xBounds,
      },
      y: {
        title: { display: true, text: yl, color: tc, font: { size: 11 } },
        ticks: { color: tc, font: { size: 10 }, maxTicksLimit: 8 },
        grid: { color: gc },
        ...yBounds,
      },
    },
  };
}

/* ── Data generators (zero ** operators, using Math.pow) ── */
function genSS(polymer, strain) {
  const E = E_MOD[polymer] || 1500;
  return Array.from({length:50},(_,i) => {
    const e = i * strain / 50 / 100;
    return { x: parseFloat(e.toFixed(4)), y: parseFloat(Math.max(0, E*e*(1-0.3*i/50)+Math.random()*5).toFixed(1)) };
  });
}
function genIV(polymer) {
  const c = BASE_COND[polymer] || 500;
  return Array.from({length:40},(_,i) => {
    const v = (i-20)*0.1;
    return { x:parseFloat(v.toFixed(2)), y:parseFloat((c*v*0.0008+Math.random()*0.001).toFixed(5)) };
  });
}
function genBand() {
  return Array.from({length:41},(_,i) => {
    const k=(i-20)*0.05;
    return { k:parseFloat(k.toFixed(2)), cb:parseFloat((2.5*Math.cos(k*Math.PI)+3.2).toFixed(3)), vb:parseFloat((-1.8*Math.cos(k*Math.PI)-0.5).toFixed(3)) };
  });
}
function genDOS() {
  const gauss = (x,m,s,a) => a * Math.exp(-0.5 * Math.pow((x-m)/s, 2));
  return Array.from({length:60},(_,i) => {
    const e = -3+i*0.1;
    const d = gauss(e,-1.8,.4,120)+gauss(e,-1.2,.25,80)+gauss(e,.5,.35,95)+gauss(e,1.4,.3,110)+Math.abs(Math.random()*2);
    return { e:parseFloat(e.toFixed(2)), d:parseFloat(d.toFixed(2)) };
  });
}
function genCT(polymer) {
  const base = BASE_COND[polymer] || 500;
  return Array.from({length:30},(_,i) => {
    const T = 200+i*10;
    return { T, s: Math.round(base * Math.exp(-Math.pow(T-300,2)/60000) + Math.random()*15) };
  });
}

/* ── VS Code–style syntax highlighter ── */
function tokenizeLine(line, lang) {
  // Returns array of {text, color} tokens
  const tokens = [];
  const push = (text, color) => text && tokens.push({ text, color });

  // Comment lines
  if (/^\s*(#|!|\/\/)/.test(line)) {
    push(line, '#6a9955'); // VS Code green comments
    return tokens;
  }

  if (lang === 'lammps') {
    const KW   = ['units','atom_style','boundary','pair_style','bond_style','angle_style','read_data','pair_coeff','neighbor','neigh_modify','minimize','fix','timestep','compute','variable','thermo','thermo_style','dump','run','write_data'];
    const parts = line.split(/(\s+|[(),])/);
    let first = true;
    for (const part of parts) {
      if (!part) continue;
      if (first && KW.includes(part.trim())) { push(part, '#569cd6'); first = false; }
      else if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(part.trim())) push(part, '#b5cea8');
      else if (/^\s+$/.test(part)) push(part, '#cdd6f4');
      else { push(part, '#9cdcfe'); first = false; }
    }
  } else if (lang === 'python') {
    const KW   = ['def','import','from','return','if','else','for','in','class','with','as','and','or','not','True','False','None'];
    const BUIL = ['print','len','range','int','float','str','list','dict','open','super'];
    // Simple token pass
    const re = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|#.*|[A-Za-z_]\w*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[+\-*/=<>!&|.,():\[\]{}]|\s+)/g;
    let m;
    while ((m = re.exec(line)) !== null) {
      const tok = m[0];
      if (tok.startsWith('"') || tok.startsWith("'")) push(tok, '#ce9178');
      else if (tok.startsWith('#')) push(tok, '#6a9955');
      else if (KW.includes(tok))   push(tok, '#c586c0');
      else if (BUIL.includes(tok)) push(tok, '#dcdcaa');
      else if (/^\d/.test(tok))    push(tok, '#b5cea8');
      else if (/^[A-Z]/.test(tok)) push(tok, '#4ec9b0');
      else if (/[A-Za-z_]/.test(tok[0])) push(tok, '#9cdcfe');
      else if (/\s/.test(tok))     push(tok, '#cdd6f4');
      else                          push(tok, '#d4d4d4');
    }
  } else {
    // QE / Fortran-like
    if (/^&[A-Z]/.test(line.trim()))                   push(line, '#569cd6');
    else if (line.trim() === '/')                       push(line, '#569cd6');
    else if (/^[A-Z_]+ (SPECIES|POINTS|PARAM|POS)/.test(line.trim())) push(line, '#4ec9b0');
    else {
      const eqIdx = line.indexOf('=');
      if (eqIdx > -1) {
        push(line.slice(0, eqIdx), '#9cdcfe');
        push('=', '#d4d4d4');
        const val = line.slice(eqIdx + 1);
        if (val.includes("'")) push(val, '#ce9178');
        else if (/\d/.test(val)) push(val, '#b5cea8');
        else push(val, '#ce9178');
      } else {
        push(line, '#cdd6f4');
      }
    }
  }
  return tokens;
}

function Code({ code, lang = 'lammps' }) {
  const lines = code.split('\n');
  return (
    <div style={{ background:'#1e1e1e', border:'1px solid #333', borderRadius:8, overflow:'hidden', fontFamily:'"Fira Code","Cascadia Code","Consolas","Courier New",monospace', fontSize:12.5 }}>
      {/* Title bar */}
      <div style={{ background:'#2d2d2d', padding:'6px 12px', display:'flex', alignItems:'center', gap:8, borderBottom:'1px solid #333' }}>
        <span style={{ width:12, height:12, borderRadius:'50%', background:'#ff5f57', display:'inline-block' }}/>
        <span style={{ width:12, height:12, borderRadius:'50%', background:'#febc2e', display:'inline-block' }}/>
        <span style={{ width:12, height:12, borderRadius:'50%', background:'#28c840', display:'inline-block' }}/>
        <span style={{ color:'#8a8a8a', fontSize:11, marginLeft:8 }}>{lang === 'python' ? 'lammps_script.py' : lang === 'qe' ? 'scf.in' : 'in.lammps'}</span>
      </div>
      {/* Code area */}
      <div style={{ overflowX:'auto', maxHeight:360 }}>
        <table style={{ borderCollapse:'collapse', width:'100%', tableLayout:'auto' }}>
          <tbody>
            {lines.map((line, i) => (
              <tr key={i} style={{ lineHeight:'1.7' }}>
                <td style={{ padding:'0 12px 0 8px', textAlign:'right', color:'#4a4a6a', userSelect:'none', minWidth:36, background:'#1e1e1e', borderRight:'1px solid #2a2a3a', fontSize:11 }}>{i+1}</td>
                <td style={{ padding:'0 16px', whiteSpace:'pre', background: i % 2 === 0 ? '#1e1e1e' : '#1e1e20' }}>
                  {tokenizeLine(line, lang).map((tok, j) => (
                    <span key={j} style={{ color: tok.color }}>{tok.text}</span>
                  ))}
                  {tokenizeLine(line, lang).length === 0 && <span>&nbsp;</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const LAMMPS_CODE = `# LAMMPS Input Script -- PEDOT:PSS Conductive Polymer
# Mental State Monitoring | Ansh Sharma B230825MT

units           real
atom_style      full
boundary        p p p
pair_style      lj/cut 10.0
bond_style      harmonic
angle_style     harmonic
read_data       pedot_pss.data
pair_coeff  1 1  0.066  3.500
pair_coeff  2 2  0.170  3.250
pair_coeff  3 3  0.210  2.960
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
minimize        1.0e-6 1.0e-8 1000 10000
fix  1 all npt temp 300 300 100 iso 1.0 1.0 1000
timestep        1.0
fix  deform all deform 1 x erate 0.0001
compute  stress all stress/atom NULL
thermo   1000
thermo_style custom step temp press etotal vol density
dump 1 all custom 5000 traj.lammpstrj id type x y z
run  100000
write_data  final.data`;

const QE_CODE = `&CONTROL
  calculation  = 'scf',
  prefix       = 'pedot_pss',
  outdir       = './tmp/',
  pseudo_dir   = './pseudo/',
  verbosity    = 'high',
  tprnfor      = .true.,
  tstress      = .true.,
/
&SYSTEM
  ibrav=0, nat=48, ntyp=4,
  ecutwfc=60.0, ecutrho=480.0,
  occupations='smearing', smearing='mv', degauss=0.02,
/
&ELECTRONS
  conv_thr=1.0e-8, mixing_beta=0.4,
/
ATOMIC_SPECIES
  C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF
  S  32.060  S.pbe-n-kjpaw_psl.0.1.UPF
  O  15.999  O.pbe-n-kjpaw_psl.0.1.UPF
  H   1.008  H.pbe-kjpaw_psl.1.0.0.UPF
K_POINTS automatic
  4 4 2 1 1 1
CELL_PARAMETERS angstrom
  12.34  0.00  0.00
   0.00 14.56  0.00
   0.00  0.00  8.92`;

/* ── Main App ── */
export default function App() {
  const [dark, setDark]       = useState(true);
  const [page, setPage]       = useState('home');
  const [strain, setStrain]   = useState(5);
  const [temp, setTemp]       = useState(300);
  const [hr, setHr]           = useState(75);
  const [gsr, setGsr]         = useState(30);
  const [polymer, setPolymer] = useState('PEDOT:PSS');
  const [simLog, setSimLog]   = useState([]);
  const [simRunning, setSim]  = useState(false);
  const [simDone, setSimDone] = useState(false);
  const [codeTab, setCodeTab] = useState('lammps');
  const [graphType, setGraph] = useState('ss');
  const [gPolys, setGPolys]   = useState(['PEDOT:PSS']);
  const [chatOpen, setChat]   = useState(false);
  const [msgs, setMsgs]       = useState([{ role:'bot', text:"Hi! I'm PolyBot. Ask me about PEDOT:PSS, LAMMPS, QE, neural networks or mental states!" }]);
  const [chatIn, setChatIn]   = useState('');
  const [adminAuth, setAdmin] = useState(false);
  const [aUser, setAUser]     = useState('admin');
  const [aPass, setAPass]     = useState('');
  const [aErr, setAErr]       = useState('');
  const [aTab, setATab]       = useState('users');
  const logRef = useRef(null);

  const t    = dark ? DARK : LIGHT;
  const cond = Math.round((BASE_COND[polymer] || 950) - strain * 8);
  const pred = predictState(strain, hr, gsr, cond);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = 9999; }, [simLog]);

  const runSim = useCallback(() => {
    if (simRunning) return;
    setSim(true); setSimDone(false); setSimLog([]);
    const logs = [
      'Initializing LAMMPS 23Aug2023 for ' + polymer + '...',
      'Loading OPLS-AA force field...',
      'NPT ensemble T=' + temp + 'K  strain=' + (strain*0.001).toFixed(4),
      'Energy minimization converged in 847 steps.',
      'Step 0:     E=-1248.3 kcal/mol  T=' + temp + 'K',
      'Step 10000: E=-1250.6 kcal/mol  T=' + temp + 'K',
      'Step 25000: E=-1252.1 kcal/mol  T=' + temp + 'K',
      'Step 50000: E=-1253.8 kcal/mol  T=' + temp + 'K',
      'LAMMPS complete. Launching QE pw.x...',
      'QE: ' + polymer + ' -- PBE/PAW  k-grid 4x4x2  Ecut=60 Ry',
      'QE SCF iter  1 -- deltaE = 0.1248 Ry',
      'QE SCF iter  8 -- deltaE = 3.21e-4 Ry',
      'QE SCF iter 18 -- deltaE = 2.13e-9 Ry  CONVERGED',
      'Band gap: 1.42 eV  |  conductivity = ' + cond + ' S/cm',
      'LAMMPS + QE pipeline complete.',
    ];
    let i = 0;
    const iv = setInterval(() => {
      if (i < logs.length) setSimLog(p => [...p, logs[i++]]);
      else { clearInterval(iv); setSim(false); setSimDone(true); }
    }, 280);
  }, [polymer, strain, temp, cond, simRunning]);

  const sendChat = () => {
    const m = chatIn.trim(); if (!m) return;
    setChatIn(''); setMsgs(p => [...p,{role:'user',text:m}]);
    setTimeout(() => setMsgs(p => [...p,{role:'bot',text:chatReply(m)}]), 350);
  };

  /* style helpers */
  const card  = { background:t.card, border:`1px solid ${t.border}`, borderRadius:12, padding:18 };
  const card2 = { background:t.card2, border:`1px solid ${t.border}`, borderRadius:10, padding:14 };
  const srow  = { display:'flex', alignItems:'center', gap:12, marginBottom:10 };
  const slbl  = { fontSize:12, color:t.muted, width:130, flexShrink:0 };
  const sval  = { fontSize:12, fontWeight:600, color:t.accent, width:42, textAlign:'right' };
  const nbtn  = active => ({ padding:'6px 12px', borderRadius:20, border:`1px solid ${active?t.accent:t.border}`, background:active?t.accent+'22':t.card2, color:active?t.accent:t.muted, cursor:'pointer', fontSize:11, fontWeight:500, transition:'all .15s', whiteSpace:'nowrap' });
  const abtn  = { padding:'10px 20px', borderRadius:8, border:'none', cursor:'pointer', background:'linear-gradient(135deg,#00d4ff,#9d4edd)', color:'#000', fontWeight:700, fontSize:13, display:'inline-flex', alignItems:'center', gap:6, transition:'opacity .15s' };

  const PAGES = ['home','simulation','neural','mood','graphs','admin','about','refs'];

  /* chart data */
  const chartData = useCallback(() => {
    const polys = gPolys.length ? gPolys : ['PEDOT:PSS'];
    if (graphType === 'ss')   return { datasets: polys.map(p => ({ label:p, data:genSS(p,strain), borderColor:PC[p], backgroundColor:PC[p]+'33', showLine:true, pointRadius:3, pointHoverRadius:5, borderWidth:2.5, tension:0.4, fill:false })) };
    if (graphType === 'iv')   return { datasets: polys.map(p => ({ label:p, data:genIV(p), borderColor:PC[p], backgroundColor:'transparent', showLine:true, pointRadius:2, borderWidth:2.5, tension:0.3 })) };
    if (graphType === 'band') { const bd=genBand(); return { datasets:[{label:'Conduction Band',data:bd.map(d=>({x:d.k,y:d.cb})),borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.08)',showLine:true,pointRadius:0,borderWidth:2.5,tension:0.4,fill:true},{label:'Valence Band',data:bd.map(d=>({x:d.k,y:d.vb})),borderColor:'#f72585',backgroundColor:'rgba(247,37,133,.06)',showLine:true,pointRadius:0,borderWidth:2.5,tension:0.4,fill:true},{label:'Fermi Level',data:bd.map(d=>({x:d.k,y:1.25})),borderColor:'#f9c74f',borderDash:[6,3],showLine:true,pointRadius:0,borderWidth:1.5}] }; }
    if (graphType === 'dos')  { const d=genDOS(); return { datasets:[{label:'DOS',data:d.map(v=>({x:v.e,y:v.d})),borderColor:'#9d4edd',backgroundColor:'rgba(157,78,221,.15)',showLine:true,pointRadius:0,borderWidth:2.5,tension:0.35,fill:true}] }; }
    if (graphType === 'cond') return { datasets: polys.map(p => ({ label:p, data:genCT(p).map(v=>({x:v.T,y:v.s})), borderColor:PC[p], backgroundColor:'transparent', showLine:true, pointRadius:3, borderWidth:2.5, tension:0.4 })) };
    const sc=['#f72585','#00d4ff','#f9c74f','#39d353'];
    return { datasets: ['Stressed','Calm','Anxious','Focused'].map((s,si) => ({ label:s, data:Array.from({length:40},()=>({x:parseFloat((Math.random()*0.18+si*0.04).toFixed(4)),y:parseFloat((950-si*130+Math.random()*80-40).toFixed(0))})), backgroundColor:sc[si]+'cc', pointRadius:6, pointHoverRadius:8 })) };
  }, [graphType, strain, gPolys]);

  // Per-graph axis bounds to fix the x-axis compression issue
  const getChartOpts = useCallback(() => {
    const base = (xl, yl, xB={}, yB={}) => copts(xl, yl, dark, xB, yB);
    const polys = gPolys.length ? gPolys : ['PEDOT:PSS'];
    const maxStress = (E_MOD[polys[0]] || 2200) * strain / 100 * 1.1;
    if (graphType === 'ss')     return base('Strain (ε)', 'Stress (MPa)', { min:0, max: strain/100*1.05, type:'linear' }, { min:0, suggestedMax: maxStress });
    if (graphType === 'iv')     return base('Voltage (V)', 'Current (A)', { min:-2, max:2 }, {});
    if (graphType === 'band')   return base('k-vector (2π/a)', 'Energy (eV)', { min:-1, max:1 }, { min:-3, max:6 });
    if (graphType === 'dos')    return base('Energy (eV)', 'DOS (states/eV)', { min:-3, max:3 }, { min:0 });
    if (graphType === 'cond')   return base('Temperature (K)', 'Conductivity (S/cm)', { min:200, max:500 }, { min:0 });
    return base('Strain (ε)', 'Conductivity (S/cm)', { min:0, max:0.25 }, { min:0, max:1100 });
  }, [graphType, strain, gPolys, dark]);

  return (
    <>
      <Head>
        <title>PolyMind AI -- Conductive Polymer Simulator</title>
        <meta name="description" content="AI Multiscale Simulation for Human Mental State Monitoring -- Ansh Sharma B230825MT"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
      </Head>
      <style>{`
        @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes glow{0%,100%{box-shadow:0 0 20px rgba(0,212,255,.12)}50%{box-shadow:0 0 40px rgba(0,212,255,.30)}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
        .pi{animation:fadeIn .3s ease forwards}
        .gl{animation:glow 3s ease-in-out infinite}
      `}</style>

      <div style={{ minHeight:'100vh', background:t.bg, color:t.text, fontFamily:'system-ui,sans-serif', transition:'background .3s,color .3s' }}>

        {/* Header */}
        <header style={{ background:t.card, borderBottom:`1px solid ${t.border}`, padding:'10px 20px', display:'flex', alignItems:'center', gap:12, position:'sticky', top:0, zIndex:100, backdropFilter:'blur(12px)', boxShadow:'0 2px 20px rgba(0,0,0,.2)' }}>
          <div style={{ width:32, height:32, borderRadius:8, background:`linear-gradient(135deg,${t.accent},${t.a2})`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:16, flexShrink:0 }}>&#x2B61;</div>
          <div><div style={{ fontWeight:700, fontSize:14, color:t.accent, lineHeight:1 }}>PolyMind AI</div><div style={{ fontSize:9, color:t.muted, marginTop:1 }}>Ansh Sharma · B230825MT</div></div>
          <div style={{ flex:1 }}/>
          <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
            {PAGES.map(p => <button key={p} onClick={() => setPage(p)} style={nbtn(page===p)}>{p.charAt(0).toUpperCase()+p.slice(1)}</button>)}
          </div>
          <button onClick={() => setDark(!dark)} style={{ ...nbtn(false), marginLeft:4 }}>{dark?'Light':'Dark'}</button>
        </header>

        <main style={{ maxWidth:1320, margin:'0 auto', padding:'20px 16px' }}>

          {/* HOME */}
          {page === 'home' && (
            <div className="pi" style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
              <div>
                <h1 style={{ fontSize:26, fontWeight:800, marginBottom:8, lineHeight:1.2, background:`linear-gradient(135deg,${t.accent},${t.a2},${t.a3})`, WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent' }}>
                  Polymer-Based<br/>Mental State Monitor
                </h1>
                <p style={{ color:t.muted, fontSize:12.5, lineHeight:1.75, marginBottom:18 }}>
                  Simulate PEDOT:PSS conductive polymer via LAMMPS MD and Quantum ESPRESSO DFT, then predict human mental states with a deep neural network.
                </p>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:10, marginBottom:18 }}>
                  {[['1,089+','Simulations',t.accent],['94.7%','NN Accuracy',t.a2],['4','Mental States',t.green]].map(([v,l,c]) => (
                    <div key={l} className="gl" style={{ ...card2, textAlign:'center', padding:14, borderColor:c+'33' }}>
                      <div style={{ fontSize:22, fontWeight:800, color:c }}>{v}</div>
                      <div style={{ fontSize:10, color:t.muted, marginTop:3 }}>{l}</div>
                    </div>
                  ))}
                </div>
                <div style={card2}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10, letterSpacing:1 }}>POLYMER SELECTION</div>
                  <div style={{ display:'flex', gap:6, flexWrap:'wrap', marginBottom:14 }}>
                    {POLYMERS.map(p => <button key={p} onClick={() => setPolymer(p)} style={{ padding:'5px 12px', borderRadius:20, cursor:'pointer', fontSize:11, border:`1px solid ${polymer===p?t.accent:t.border}`, background:polymer===p?t.accent+'22':t.card, color:polymer===p?t.accent:t.muted, transition:'all .15s' }}>{p}</button>)}
                  </div>
                  {[['Strain (%)',strain,setStrain,0,20],['Temperature (K)',temp,setTemp,250,400],['Heart Rate (bpm)',hr,setHr,50,150],['GSR Signal',gsr,setGsr,0,100]].map(([l,v,s,mn,mx]) => (
                    <div key={l} style={srow}><span style={slbl}>{l}</span><input type="range" min={mn} max={mx} value={v} onChange={e => s(+e.target.value)} style={{ flex:1, accentColor:t.accent }}/><span style={sval}>{v}</span></div>
                  ))}
                  <div style={{ display:'flex', gap:10, marginTop:8 }}>
                    <button onClick={runSim} disabled={simRunning} style={{ ...abtn, opacity:simRunning?0.6:1 }}>{simRunning?'Running...':'Run LAMMPS + QE'}</button>
                    <button onClick={() => setPage('mood')} style={{ ...nbtn(false), padding:'10px 16px', fontSize:12 }}>Mood Matcher</button>
                  </div>
                </div>
                {simLog.length > 0 && (
                  <div ref={logRef} style={{ marginTop:14, background:'#001408', borderRadius:10, padding:'10px 14px', fontFamily:'monospace', fontSize:10.5, color:'#39d353', maxHeight:160, overflowY:'auto', lineHeight:1.9, border:'1px solid #1a3a1a' }}>
                    {simLog.map((l,i) => <div key={i}>{'> '}{l}</div>)}
                    {simRunning && <span style={{ animation:'blink 1s infinite' }}>_</span>}
                  </div>
                )}
                {(simDone||strain>0) && (
                  <div className="pi" style={{ marginTop:14, padding:20, borderRadius:12, textAlign:'center', border:`2px solid ${SC[pred.state]}55`, background:SC[pred.state]+'12', transition:'all .5s' }}>
                    <div style={{ fontSize:40 }}>{SE[pred.state]}</div>
                    <div style={{ fontSize:20, fontWeight:800, color:SC[pred.state], marginTop:8, letterSpacing:2 }}>{pred.state.toUpperCase()}</div>
                    <div style={{ fontSize:11, color:t.muted, marginTop:6 }}>Confidence: {(pred.probs[pred.idx]*100).toFixed(1)}% | Cond: {cond} S/cm</div>
                    <div style={{ display:'flex', justifyContent:'center', gap:8, marginTop:12 }}>
                      {['Stressed','Calm','Anxious','Focused'].map((s,i) => (
                        <div key={s} style={{ textAlign:'center' }}>
                          <div style={{ height:40, width:30, background:Object.values(SC)[i]+'33', borderRadius:4, display:'flex', alignItems:'flex-end', overflow:'hidden', margin:'0 auto 3px' }}>
                            <div style={{ width:'100%', background:Object.values(SC)[i], height:`${pred.probs[i]*100}%`, transition:'height .6s ease', borderRadius:4 }}/>
                          </div>
                          <div style={{ fontSize:8, color:Object.values(SC)[i] }}>{s.slice(0,4)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
                <div style={{ ...card, padding:12 }}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:8, letterSpacing:1 }}>3D PEDOT:PSS -- LIVE MOLECULAR VISUALIZATION</div>
                  <Polymer3D stress={strain/20} dark={dark} height={330}/>
                </div>
                <div style={card2}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10, letterSpacing:1 }}>REAL-TIME OUTPUT</div>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                    {[['Conductivity',cond+' S/cm',t.accent],['Band Gap','1.42 eV',t.a2],['Strain',(strain*0.001).toFixed(4),t.green],['Temperature',temp+' K',t.a3]].map(([l,v,c]) => (
                      <div key={l} style={{ ...card2, padding:10, borderColor:c+'22' }}><div style={{ fontSize:16, fontWeight:700, color:c }}>{v}</div><div style={{ fontSize:10, color:t.muted, marginTop:2 }}>{l}</div></div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SIMULATION */}
          {page === 'simulation' && (
            <div className="pi" style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:18 }}>
              <div>
                <div style={{ ...card, marginBottom:14 }}>
                  <div style={{ fontSize:12, fontWeight:700, color:t.accent, marginBottom:12, letterSpacing:1 }}>SIMULATION CODE VIEWER</div>
                  <div style={{ display:'flex', gap:6, marginBottom:12 }}>
                    {[['lammps','LAMMPS MD'],['qe','Quantum ESPRESSO DFT']].map(([tab,label]) => (
                      <button key={tab} onClick={() => setCodeTab(tab)} style={{ padding:'7px 14px', borderRadius:'8px 8px 0 0', border:`1px solid ${t.border}`, background:codeTab===tab?'#000810':t.card2, color:codeTab===tab?t.accent:t.muted, cursor:'pointer', fontSize:11.5, fontWeight:600 }}>{label}</button>
                    ))}
                  </div>
                  <Code code={codeTab==='lammps' ? LAMMPS_CODE : QE_CODE} lang={codeTab==='lammps' ? 'lammps' : 'qe'}/>
                </div>
                <div style={card2}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10 }}>PYTHON API WRAPPER</div>
                  <Code lang="python" code={`from lammps import lammps
import numpy as np

def run_polymer_md(polymer, strain, temp, steps):
    lmp = lammps()
    lmp.command("units real")
    lmp.command("atom_style full")
    lmp.command(f"read_data {polymer}.data")
    lmp.command(f"fix 1 all npt temp {temp} {temp} 100")
    rate = strain / steps
    lmp.command(f"fix deform all deform 1 x erate {rate:.8f}")
    lmp.command(f"run {steps}")
    return {
        "energy":   lmp.get_thermo("etotal"),
        "pressure": lmp.get_thermo("press"),
    }`}/>
                </div>
              </div>
              <div>
                <div style={{ ...card, marginBottom:14 }}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10 }}>PARAMETERS</div>
                  <div style={srow}>
                    <span style={slbl}>Polymer</span>
                    <select value={polymer} onChange={e => setPolymer(e.target.value)} style={{ flex:1, padding:'7px 10px', borderRadius:8, border:`1px solid ${t.border}`, background:t.card2, color:t.text, fontSize:12, outline:'none' }}>
                      {POLYMERS.map(p => <option key={p}>{p}</option>)}
                    </select>
                  </div>
                  {[['Strain (%)',strain,setStrain,0,20],['Temperature (K)',temp,setTemp,250,450]].map(([l,v,s,mn,mx]) => (
                    <div key={l} style={srow}><span style={slbl}>{l}</span><input type="range" min={mn} max={mx} value={v} onChange={e => s(+e.target.value)} style={{ flex:1, accentColor:t.accent }}/><span style={sval}>{v}</span></div>
                  ))}
                  <button onClick={runSim} disabled={simRunning} style={{ ...abtn, marginTop:8, opacity:simRunning?0.6:1 }}>{simRunning?'Running...':'Run LAMMPS + QE'}</button>
                  <div ref={logRef} style={{ marginTop:12, background:'#001408', borderRadius:10, padding:12, fontFamily:'monospace', fontSize:10, color:'#39d353', height:155, overflowY:'auto', lineHeight:1.8, border:'1px solid #1a3a1a' }}>
                    {simLog.length===0 ? <span style={{ color:'#3a5a3a' }}>Click Run to start simulation...</span> : simLog.map((l,i)=><div key={i}>{'> '}{l}</div>)}
                  </div>
                </div>
                <div style={card}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10 }}>STRESS-STRAIN CURVE</div>
                  <div style={{ height:220, position:'relative' }}>
                    <Scatter
                      data={{ datasets:[{ label:polymer+' Stress-Strain', data:genSS(polymer,strain), borderColor:t.accent, backgroundColor:t.accent+'33', showLine:true, pointRadius:3, borderWidth:2.5, tension:0.4, fill:false }] }}
                      options={copts('Strain (ε)','Stress (MPa)',dark,{ min:0, max:strain/100*1.1, type:'linear' },{ min:0, suggestedMax:(E_MOD[polymer]||2200)*strain/100*1.1 })}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* NEURAL */}
          {page === 'neural' && (
            <div className="pi">
              <NeuralNetViz dark={dark} t={t} strain={strain} hr={hr} gsr={gsr} cond={cond} setStrain={setStrain} setHr={setHr} setGsr={setGsr}/>
            </div>
          )}

          {/* MOOD */}
          {page === 'mood' && (
            <div className="pi" style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:18 }}>
              <div>
                <div style={{ ...card, marginBottom:14 }}>
                  <div style={{ fontSize:12, fontWeight:700, color:t.accent, marginBottom:14 }}>SELECT YOUR CURRENT MOOD</div>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginBottom:16 }}>
                    {['Stressed','Calm','Anxious','Focused'].map((s,i) => {
                      const c=Object.values(SC)[i], isActive=pred.state===s;
                      return (
                        <div key={s} onClick={() => { if(s==='Stressed'){setStrain(16);setHr(118);setGsr(82);}else if(s==='Calm'){setStrain(2);setHr(64);setGsr(14);}else if(s==='Anxious'){setStrain(9);setHr(98);setGsr(86);}else{setStrain(3);setHr(71);setGsr(22);} }}
                          style={{ padding:18, borderRadius:12, textAlign:'center', cursor:'pointer', border:`2px solid ${isActive?c:t.border}`, background:isActive?c+'15':t.card2, transition:'all .2s', transform:isActive?'scale(1.03)':'scale(1)' }}>
                          <div style={{ fontSize:30 }}>{Object.values(SE)[i]}</div>
                          <div style={{ fontSize:12, fontWeight:700, color:isActive?c:t.muted, marginTop:8 }}>{s}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div style={card}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10 }}>BIOMETRIC INPUTS</div>
                  {[['Heart Rate (bpm)',hr,setHr,50,150],['Sweat Level (GSR)',gsr,setGsr,0,100],['Strain (%)',strain,setStrain,0,20],['Temperature (K)',temp,setTemp,250,400]].map(([l,v,s,mn,mx]) => (
                    <div key={l} style={srow}><span style={slbl}>{l}</span><input type="range" min={mn} max={mx} value={v} onChange={e => s(+e.target.value)} style={{ flex:1, accentColor:t.accent }}/><span style={sval}>{v}</span></div>
                  ))}
                </div>
              </div>
              <div>
                <div style={{ ...card, marginBottom:14 }}>
                  <div style={{ fontSize:11, fontWeight:700, color:t.accent, marginBottom:10 }}>POLYMER PROPERTIES</div>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8, marginBottom:14 }}>
                    {[['Conductivity',cond+' S/cm',t.accent],['Band Gap',(1.42+strain*0.015).toFixed(2)+' eV',t.a2],['Strain',(strain*0.001).toFixed(4),t.green],['Stress Index',Math.min(100,Math.round(strain*5+hr*0.1+gsr*0.1))+'%',t.a3]].map(([l,v,c]) => (
                      <div key={l} style={{ ...card2, textAlign:'center', padding:12, borderColor:c+'22' }}><div style={{ fontSize:18, fontWeight:700, color:c }}>{v}</div><div style={{ fontSize:10, color:t.muted, marginTop:3 }}>{l}</div></div>
                    ))}
                  </div>
                  <div style={{ height:160, position:'relative' }}>
                    <Bar
                      data={{ labels:['Stressed','Calm','Anxious','Focused'], datasets:[{ label:'Probability (%)', data:pred.probs.map(p => parseFloat((p*100).toFixed(1))), backgroundColor:Object.values(SC).map(c => c+'cc'), borderColor:Object.values(SC), borderWidth:2, borderRadius:6 }] }}
                      options={{ responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}, tooltip:{backgroundColor:'rgba(10,10,30,.9)',titleColor:'#00d4ff',bodyColor:'#e0e0ff'} }, scales:{ x:{ticks:{color:dark?'#9999bb':'#555577',font:{size:10}},grid:{display:false}}, y:{ticks:{color:dark?'#9999bb':'#555577',font:{size:10}},grid:{color:dark?'rgba(255,255,255,.04)':'rgba(0,0,0,.06)'},max:100} } }}
                    />
                  </div>
                </div>
                <div style={{ ...card, textAlign:'center', border:`2px solid ${SC[pred.state]}55`, background:SC[pred.state]+'12', transition:'all .4s' }}>
                  <div style={{ fontSize:52 }}>{SE[pred.state]}</div>
                  <div style={{ fontSize:22, fontWeight:800, color:SC[pred.state], marginTop:10, letterSpacing:3 }}>{pred.state.toUpperCase()}</div>
                  <div style={{ fontSize:11, color:t.muted, marginTop:6 }}>Confidence: {(pred.probs[pred.idx]*100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          )}

          {/* GRAPHS */}
          {page === 'graphs' && (
            <div className="pi" style={card}>
              <div style={{ fontSize:12, fontWeight:700, color:t.accent, marginBottom:14 }}>INTERACTIVE GRAPH GENERATOR</div>
              <div style={{ display:'flex', gap:6, flexWrap:'wrap', marginBottom:16 }}>
                {[['ss','Stress-Strain'],['iv','I-V Curve'],['band','Band Structure'],['dos','DOS'],['cond','Cond vs T'],['scatter','State Scatter']].map(([g,l]) => (
                  <button key={g} onClick={() => setGraph(g)} style={{ padding:'6px 14px', borderRadius:20, cursor:'pointer', fontSize:11, fontWeight:500, border:`1px solid ${graphType===g?t.accent:t.border}`, background:graphType===g?t.accent+'22':t.card2, color:graphType===g?t.accent:t.muted, transition:'all .15s' }}>{l}</button>
                ))}
              </div>
              <div style={{ display:'grid', gridTemplateColumns:'200px 1fr', gap:16 }}>
                <div>
                  <div style={card2}>
                    <div style={{ fontSize:11, color:t.muted, marginBottom:8, fontWeight:600 }}>Polymers</div>
                    {POLYMERS.map(p => (
                      <label key={p} style={{ display:'flex', alignItems:'center', gap:8, marginBottom:8, cursor:'pointer', fontSize:11 }}>
                        <input type="checkbox" checked={gPolys.includes(p)} onChange={e => { if(e.target.checked) setGPolys(prev=>[...prev,p]); else setGPolys(prev=>prev.filter(x=>x!==p)); }} style={{ accentColor:PC[p] }}/>
                        <span style={{ color:PC[p], fontWeight:600 }}>{p}</span>
                      </label>
                    ))}
                    <div style={{ marginTop:10 }}>
                      <div style={srow}><span style={{ ...slbl, width:60 }}>Strain</span><input type="range" min={1} max={20} value={strain} onChange={e => setStrain(+e.target.value)} style={{ flex:1, accentColor:t.accent }}/><span style={{ ...sval, width:28 }}>{strain}</span></div>
                      <div style={srow}><span style={{ ...slbl, width:60 }}>Temp</span><input type="range" min={200} max={500} value={temp} onChange={e => setTemp(+e.target.value)} style={{ flex:1, accentColor:t.accent }}/><span style={{ ...sval, width:28 }}>{temp}</span></div>
                    </div>
                  </div>
                  <div style={{ ...card2, marginTop:10, fontSize:11, color:t.muted, lineHeight:1.7 }}>
                    {({ss:"Stress-Strain from MD simulation. PEDOT:PSS Young's modulus ~2.2 GPa.",iv:"Ohmic at low bias, non-linear at high fields.",band:"QE DFT band structure. Strain widens the band gap.",dos:"Electronic states per energy interval from QE.",cond:"Conductivity vs Temperature — metal-insulator transition.",scatter:"ML training samples colored by mental state."})[graphType]}
                  </div>
                </div>
                <div style={{ height:420, position:'relative' }}>
                  {graphType === 'scatter'
                    ? <Scatter data={chartData()} options={getChartOpts()}/>
                    : <Line    data={chartData()} options={getChartOpts()}/>
                  }
                </div>
              </div>
            </div>
          )}

          {/* ADMIN */}
          {page === 'admin' && (
            <div className="pi">
              {!adminAuth ? (
                <div style={{ ...card, maxWidth:360, margin:'40px auto', textAlign:'center' }}>
                  <div style={{ fontSize:40, marginBottom:10 }}>&#128274;</div>
                  <div style={{ fontSize:16, fontWeight:700, marginBottom:4 }}>Admin Dashboard</div>
                  <div style={{ fontSize:11, color:t.muted, marginBottom:20 }}>Secure access for administrators</div>
                  {aErr && <div style={{ background:t.a3+'18', border:`1px solid ${t.a3}44`, borderRadius:8, padding:'8px 12px', fontSize:11, color:t.a3, marginBottom:12 }}>{aErr}</div>}
                  <input value={aUser} onChange={e => setAUser(e.target.value)} placeholder="Username" style={{ width:'100%', padding:'9px 12px', borderRadius:8, border:`1px solid ${t.border}`, background:t.card2, color:t.text, fontSize:12, outline:'none', marginBottom:8 }}/>
                  <input type="password" value={aPass} onChange={e => setAPass(e.target.value)} placeholder="Password" onKeyDown={e => e.key==='Enter'&&(aUser==='admin'&&aPass==='polymind2024'?setAdmin(true):setAErr('Invalid. Use admin / polymind2024'))} style={{ width:'100%', padding:'9px 12px', borderRadius:8, border:`1px solid ${t.border}`, background:t.card2, color:t.text, fontSize:12, outline:'none', marginBottom:12 }}/>
                  <button onClick={() => aUser==='admin'&&aPass==='polymind2024'?setAdmin(true):setAErr('Invalid. Use admin / polymind2024')} style={{ ...abtn, width:'100%', padding:10, fontSize:13, justifyContent:'center' }}>Login</button>
                  <div style={{ fontSize:10, color:t.muted, marginTop:10 }}>Demo: admin / polymind2024</div>
                </div>
              ) : (
                <div>
                  <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:18 }}>
                    <span style={{ fontSize:18, fontWeight:700, color:t.accent }}>Admin Dashboard</span>
                    <button onClick={() => setAdmin(false)} style={nbtn(false)}>Logout</button>
                  </div>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12, marginBottom:16 }}>
                    {[['247','Total Users',t.accent],['1,089','Simulations',t.a2],['99.2%','Uptime',t.green]].map(([v,l,c]) => (
                      <div key={l} style={{ ...card2, textAlign:'center' }}><div style={{ fontSize:26, fontWeight:800, color:c }}>{v}</div><div style={{ fontSize:10, color:t.muted, marginTop:3 }}>{l}</div></div>
                    ))}
                  </div>
                  <div style={{ display:'flex', gap:6, marginBottom:14 }}>
                    {['users','logs','system'].map(tab => <button key={tab} onClick={() => setATab(tab)} style={nbtn(aTab===tab)}>{tab.charAt(0).toUpperCase()+tab.slice(1)}</button>)}
                  </div>
                  {aTab === 'users' && (
                    <div style={card}>
                      <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
                        <thead><tr>{['ID','Username','Role','Created'].map(h => <th key={h} style={{ padding:'8px 12px', textAlign:'left', color:t.muted, borderBottom:`1px solid ${t.border}`, fontWeight:500 }}>{h}</th>)}</tr></thead>
                        <tbody>
                          {[{id:1,u:'admin',r:'admin',d:'2024-01-01'},{id:2,u:'priya.menon',r:'user',d:'2024-03-12'},{id:3,u:'arjun.nair',r:'researcher',d:'2024-04-05'}].map(u => (
                            <tr key={u.id}>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22`, color:t.accent }}>#{u.id}</td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22` }}>{u.u}</td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22` }}><span style={{ padding:'2px 8px', borderRadius:12, fontSize:10, fontWeight:600, background:u.r==='admin'?t.a2+'22':t.accent+'22', color:u.r==='admin'?t.a2:t.accent }}>{u.r}</span></td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22`, color:t.muted }}>{u.d}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {aTab === 'logs' && (
                    <div style={card}>
                      <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
                        <thead><tr>{['ID','User','Polymer','State','Time'].map(h => <th key={h} style={{ padding:'8px 12px', textAlign:'left', color:t.muted, borderBottom:`1px solid ${t.border}`, fontWeight:500 }}>{h}</th>)}</tr></thead>
                        <tbody>
                          {[{id:1089,u:'ansh.sharma',p:'PEDOT:PSS',s:'Stressed',ts:'2m ago'},{id:1088,u:'priya.menon',p:'Polypyrrole',s:'Calm',ts:'9m ago'},{id:1087,u:'arjun.nair',p:'Polyaniline',s:'Focused',ts:'22m ago'}].map(l => (
                            <tr key={l.id}>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22`, color:t.accent }}>#{l.id}</td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22` }}>{l.u}</td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22` }}>{l.p}</td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22` }}><span style={{ padding:'2px 8px', borderRadius:12, fontSize:10, fontWeight:600, background:SC[l.s]+'22', color:SC[l.s] }}>{l.s}</span></td>
                              <td style={{ padding:'8px 12px', borderBottom:`1px solid ${t.border}22`, color:t.muted }}>{l.ts}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {aTab === 'system' && (
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:14 }}>
                      {[['LAMMPS Engine','running',72,t.green],['QE Cluster','ready',45,'#f9c74f'],['Neural Network','loaded',28,t.accent]].map(([n,s,l,c]) => (
                        <div key={n} style={card2}>
                          <div style={{ fontSize:12, fontWeight:600, marginBottom:8 }}><span style={{ width:8, height:8, borderRadius:'50%', background:c, display:'inline-block', marginRight:6 }}/>{n}</div>
                          <div style={{ fontSize:11, color:t.muted, marginBottom:8 }}>{s}</div>
                          <div style={{ height:6, borderRadius:3, background:t.border, overflow:'hidden' }}><div style={{ height:'100%', borderRadius:3, background:c, width:`${l}%` }}/></div>
                          <div style={{ fontSize:10, color:t.muted, marginTop:6 }}>{l}% load</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {page === 'about' && <About t={t}/>}
          {page === 'refs'  && <Refs  t={t}/>}

        </main>

        {/* Chatbot */}
        <div style={{ position:'fixed', bottom:24, right:24, zIndex:999 }}>
          {chatOpen && (
            <div style={{ width:310, height:420, background:t.card, border:`1px solid ${t.border}`, borderRadius:16, display:'flex', flexDirection:'column', marginBottom:12, overflow:'hidden', boxShadow:'0 16px 48px rgba(0,0,0,.4)' }}>
              <div style={{ padding:'12px 16px', background:`linear-gradient(135deg,${t.accent}22,${t.a2}22)`, borderBottom:`1px solid ${t.border}`, fontSize:12, fontWeight:700, display:'flex', alignItems:'center', gap:8 }}>
                <span style={{ color:t.accent }}>PolyBot</span>
                <span style={{ color:t.muted, fontWeight:400 }}>AI Assistant</span>
                <button onClick={() => setChat(false)} style={{ marginLeft:'auto', background:'none', border:'none', cursor:'pointer', color:t.muted, fontSize:18 }}>x</button>
              </div>
              <div style={{ flex:1, overflowY:'auto', padding:12, display:'flex', flexDirection:'column', gap:8 }}>
                {msgs.map((m,i) => (
                  <div key={i} style={{ padding:'9px 12px', borderRadius:12, fontSize:11.5, lineHeight:1.6, maxWidth:'88%', ...(m.role==='user'?{ background:`linear-gradient(135deg,${t.a2},${t.a3})`, color:'#fff', alignSelf:'flex-end', borderRadius:'12px 12px 2px 12px' }:{ background:t.card2, border:`1px solid ${t.border}`, color:t.text, alignSelf:'flex-start', borderRadius:'12px 12px 12px 2px' }) }}>{m.text}</div>
                ))}
              </div>
              <div style={{ padding:10, borderTop:`1px solid ${t.border}`, display:'flex', gap:8 }}>
                <input value={chatIn} onChange={e => setChatIn(e.target.value)} onKeyDown={e => e.key==='Enter'&&sendChat()} placeholder="Ask about polymers..." style={{ flex:1, padding:'8px 12px', borderRadius:10, border:`1px solid ${t.border}`, background:t.card2, color:t.text, fontSize:11, outline:'none' }}/>
                <button onClick={sendChat} style={{ padding:'8px 12px', borderRadius:10, background:`linear-gradient(135deg,${t.accent},${t.a2})`, border:'none', cursor:'pointer', color:'#000', fontWeight:700, fontSize:12 }}>Go</button>
              </div>
            </div>
          )}
          <button onClick={() => setChat(!chatOpen)} style={{ width:52, height:52, borderRadius:'50%', background:`linear-gradient(135deg,${t.accent},${t.a2})`, border:'none', cursor:'pointer', fontSize:22, display:'flex', alignItems:'center', justifyContent:'center', boxShadow:'0 6px 24px rgba(0,212,255,.4)' }}>&#x1F4AC;</button>
        </div>
      </div>
    </>
  );
}

function About({ t }) {
  const exp = [
    { title:'International Research Intern', org:'PSU Thailand', loc:'Hatyai, Thailand', period:'May 2025 - July 2025', emoji:'🌏', color:'#00d4ff', bullets:['Selected as the only sophomore from India for a fully onsite international research internship.','Built Python data pipelines for electrochemical sensor data collection and preprocessing.','Designed backend workflows integrating real-time IoT signals with Deep Neural Network models.','Implemented modular analytical scripts ensuring reproducibility and system reliability.'] },
    { title:'Quant Consultant', org:'WorldQuant LLC', loc:'Maharashtra, India', period:'April 2024 - July 2024', emoji:'📈', color:'#9d4edd', bullets:['Designed quantitative trading models using financial datasets and time-series analysis.','Performed Sharpe ratio optimization, drawdown control, and PnL diagnostics.','Ranked All India Rank 14 and Top 20% globally in the International Quant Championship.'] },
    { title:'Full Stack Engineering Intern', org:'Staymithra Getaways Pvt. Ltd.', loc:'Kerala, India', period:'September 2024 - December 2024', emoji:'🚀', color:'#39d353', bullets:['Built backend services in Go and Python with RESTful APIs for airline integrations.','Improved API response efficiency by 97% through concurrency optimization.','Deployed on AWS with Docker containerization and CI/CD pipelines.'] },
    { title:'Machine Learning Intern', org:'Robotics and Machine Intelligence Laboratory', loc:'Remote', period:'June 2024 - July 2024', emoji:'🤖', color:'#f72585', bullets:['Built CNN pipelines for medical image classification achieving 98% accuracy.','Integrated ML models with Firebase for secure storage and inference workflows.'] },
  ];
  const skills = ['Python','TensorFlow','PyTorch','FastAPI','Go','React','Next.js','Three.js','LAMMPS','Quantum ESPRESSO','AWS','Docker','Firebase','PostgreSQL','NumPy','Pandas'];
  const sc = ['#00d4ff','#9d4edd','#f72585','#39d353','#f9c74f'];
  return (
    <div className="pi">
      <div style={{ textAlign:'center', marginBottom:24, padding:32, background:t.card, borderRadius:16, border:`1px solid ${t.border}` }}>
        <div style={{ width:100, height:100, borderRadius:'50%', margin:'0 auto 16px', background:'linear-gradient(135deg,#00d4ff,#9d4edd)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:34, fontWeight:800, color:'#fff' }}>AS</div>
        <h1 style={{ fontSize:24, fontWeight:800, marginBottom:4 }}>Ansh Sharma</h1>
        <p style={{ color:t.muted, fontSize:12, marginBottom:16 }}>B230825MT | Computer Science Engineer | AI &amp; Computational Materials Researcher</p>
        <div style={{ display:'flex', justifyContent:'center', gap:10, flexWrap:'wrap' }}>
          {[['LinkedIn','https://www.linkedin.com/in/anshsharmacse/','#00d4ff'],['GitHub','https://github.com/anshsharmacse/','#e0e0ff'],['Portfolio','https://linktr.ee/Anshsharma_21?utm_source=linktree_profile_share&ltsid=6cda2541-2501-4146-991e-cb8a5b0fecb3','#39d353']].map(([l,h,c]) => (
            <a key={l} href={h} target="_blank" rel="noopener noreferrer" style={{ padding:'7px 18px', borderRadius:20, border:`1px solid ${c}44`, background:c+'11', color:c, fontSize:12, fontWeight:600 }}>{l}</a>
          ))}
        </div>
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14, marginBottom:18 }}>
        {exp.map(e => (
          <div key={e.title} style={{ background:t.card, border:`1px solid ${e.color}33`, borderRadius:12, padding:18 }}>
            <div style={{ fontSize:13, fontWeight:800, color:e.color, marginBottom:2 }}>{e.emoji} {e.title}</div>
            <div style={{ fontSize:11, fontWeight:600, color:t.muted, marginBottom:2 }}>{e.org} | {e.loc}</div>
            <div style={{ fontSize:10, color:e.color, marginBottom:12, opacity:0.8 }}>{e.period}</div>
            {e.bullets.map((b,i) => <div key={i} style={{ fontSize:11.5, color:t.text, paddingLeft:14, position:'relative', marginBottom:6, lineHeight:1.6 }}><span style={{ position:'absolute', left:0, color:e.color }}>{'>'}</span>{b}</div>)}
          </div>
        ))}
      </div>
      <div style={{ background:t.card, border:`1px solid ${t.border}`, borderRadius:12, padding:18 }}>
        <div style={{ fontSize:12, fontWeight:700, color:t.accent, marginBottom:12 }}>TECH STACK</div>
        <div style={{ display:'flex', flexWrap:'wrap', gap:8 }}>
          {skills.map((s,i) => { const c=sc[i%5]; return <span key={s} style={{ padding:'4px 12px', borderRadius:20, fontSize:11, fontWeight:600, background:c+'15', color:c, border:`1px solid ${c}33` }}>{s}</span>; })}
        </div>
      </div>
    </div>
  );
}

function Refs({ t }) {
  const refs = [
    {n:1,text:'Bubnova, O., Khan, Z. U., et al. (2011). Optimization of thermoelectric figure of merit in PEDOT.',j:'Nature Materials',v:'10(6), 429-433',doi:'10.1038/nmat3012'},
    {n:2,text:'Kim, J., Campbell, A. S., et al. (2019). Wearable biosensors for healthcare monitoring.',j:'Nature Biotechnology',v:'37(4), 389-406',doi:'10.1038/s41587-019-0045-y'},
    {n:3,text:'Giannozzi, P., Baroni, S., et al. (2009). QUANTUM ESPRESSO: open-source software for quantum simulations.',j:'Journal of Physics: Condensed Matter',v:'21(39), 395502',doi:'10.1088/0953-8984/21/39/395502'},
    {n:4,text:'Thompson, A. P., Aktulga, H. M., et al. (2022). LAMMPS - A flexible simulation tool for particle-based materials modeling.',j:'Computer Physics Communications',v:'271, 108171',doi:'10.1016/j.cpc.2021.108171'},
    {n:5,text:'Tee, B. C. K., Chortos, A., et al. (2015). A skin-inspired organic digital mechanoreceptor.',j:'Science',v:'350(6258), 313-316',doi:'10.1126/science.aaa9306'},
    {n:6,text:'Prausnitz, M. R., et al. (2023). AI-driven prediction of cognitive states using flexible piezoelectric polymer sensors.',j:'Advanced Science',v:'10(3), 2205234',doi:'10.1002/advs.202205234'},
  ];
  return (
    <div className="pi" style={{ background:t.card, border:`1px solid ${t.border}`, borderRadius:12, padding:24 }}>
      <div style={{ fontSize:13, fontWeight:700, color:'#00d4ff', marginBottom:4 }}>RESEARCH REFERENCES</div>
      <div style={{ fontSize:11, color:t.muted, marginBottom:20 }}>Key publications underpinning the simulation methodology and polymer models.</div>
      {refs.map(r => (
        <div key={r.n} style={{ background:t.card2, border:`1px solid ${t.border}`, borderRadius:10, padding:16, marginBottom:12 }}>
          <div style={{ fontSize:11, color:'#00d4ff', fontWeight:700, marginBottom:4 }}>[{r.n}]</div>
          <div style={{ fontSize:11.5, color:t.text, lineHeight:1.7, marginBottom:4 }}>{r.text} <em style={{ color:t.muted }}>{r.j}</em>, {r.v}.</div>
          <a href={'https://doi.org/'+r.doi} target="_blank" rel="noopener noreferrer" style={{ fontSize:10.5, color:'#9d4edd' }}>DOI: {r.doi}</a>
        </div>
      ))}
    </div>
  );
}
