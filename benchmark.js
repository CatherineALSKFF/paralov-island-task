#!/usr/bin/env node
// BENCHMARK — test model+VP configs against REAL GT on completed rounds
// No hallucinating. Just numbers.
const fs = require('fs'), path = require('path');
const H = 40, W = 40, SEEDS = 5, C = 6;
const DD = path.join(__dirname, 'data');
const log = m => console.log(`[${new Date().toISOString().slice(11,19)}] ${m}`);
function t2c(t) { return (t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0; }

// Feature extraction (same as autopilot_simple — proven)
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++) {
    if (!dy&&!dx) continue; const ny=y+dy,nx=x+dx;
    if (ny<0||ny>=H||nx<0||nx>=W) continue; const nt=g[ny][nx];
    if (nt===1||nt===2) nS++; if (nt===10) co=1; if (nt===4) fN++; }
  for (let dy=-2;dy<=2;dy++) for (let dx=-2;dx<=2;dx++) {
    if (Math.abs(dy)<=1&&Math.abs(dx)<=1) continue; const ny=y+dy,nx=x+dx;
    if (ny<0||ny>=H||nx<0||nx>=W) continue;
    if (g[ny][nx]===1||g[ny][nx]===2) sR2++; }
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3;
  const fb=fN<=1?0:fN<=3?1:2;
  return {d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`,d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    d2:`D2_${t}_${sa>0?1:0}_${co}`,d3:`D3_${t}_${co}`,d4:`D4_${t}`};
}

// Load all data
const I={},G={},R={};
for (let r=1;r<=11;r++) { const rn=`R${r}`;
  if (fs.existsSync(path.join(DD,`inits_${rn}.json`))) I[rn]=JSON.parse(fs.readFileSync(path.join(DD,`inits_${rn}.json`)));
  if (fs.existsSync(path.join(DD,`gt_${rn}.json`))) G[rn]=JSON.parse(fs.readFileSync(path.join(DD,`gt_${rn}.json`)));
  if (fs.existsSync(path.join(DD,`replays_${rn}.json`))) R[rn]=JSON.parse(fs.readFileSync(path.join(DD,`replays_${rn}.json`)));
}

// VP data for completed rounds
const VP_ROUNDS = {
  R8: JSON.parse(fs.readFileSync(path.join(DD,'viewport_c5cdf100.json'))),
  R9: JSON.parse(fs.readFileSync(path.join(DD,'viewport_2a341ace.json'))),
  R10: JSON.parse(fs.readFileSync(path.join(DD,'viewport_75e625c3.json'))),
};

log(`Data: ${Object.keys(I).length} inits, ${Object.keys(G).length} GTs, ${Object.keys(R).length} replay sets`);
log(`VP rounds: ${Object.keys(VP_ROUNDS).join(', ')} (${Object.values(VP_ROUNDS).map(v=>v.length+' obs').join(', ')})`);

// Build model excluding one round
function buildModel(excludeRn, useRoundWeight) {
  const TR = Object.keys(I).filter(k => G[k] && k !== excludeRn);
  const model = {};
  for (const level of ['d0','d1','d2','d3','d4']) {
    const m = {};
    for (const rn of TR) { if (!G[rn]||!I[rn]) continue;
      const rw = useRoundWeight ? Math.pow(1.1, parseInt(rn.replace('R',''))-1) : 1;
      for (let si=0;si<SEEDS;si++) { if (!I[rn][si]||!G[rn][si]) continue;
        for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
          const keys=cf(I[rn][si],y,x); if (!keys) continue; const k=keys[level];
          if (!m[k]) m[k]={n:0,counts:new Float64Array(C)};
          const p=G[rn][si][y][x]; const gtW=20*rw;
          for (let c=0;c<C;c++) m[k].counts[c]+=p[c]*gtW; m[k].n+=gtW; } } }
    for (const rn of TR) { if (!R[rn]||!I[rn]) continue;
      const rw = useRoundWeight ? Math.pow(1.1, parseInt(rn.replace('R',''))-1) : 1;
      for (const rep of R[rn]) { const g=I[rn][rep.si]; if (!g) continue;
        for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
          const keys=cf(g,y,x); if (!keys) continue; const k=keys[level];
          const fc=t2c(rep.finalGrid[y][x]);
          if (!m[k]) m[k]={n:0,counts:new Float64Array(C)}; m[k].n+=rw; m[k].counts[fc]+=rw; } } }
    for (const k of Object.keys(m)) {
      const tot=Array.from(m[k].counts).reduce((a,b)=>a+b,0)+C*0.05;
      m[k].a=Array.from(m[k].counts).map(v=>(v+0.05)/tot); }
    for (const [k,v] of Object.entries(m)) { if (!model[k]) model[k]=v; }
  }
  return model;
}

// VP fusion
function fuseVP(model, vpObs, inits, CW) {
  const vpD0={};
  for (const obs of vpObs) {
    for (let dy=0;dy<obs.grid.length;dy++) for (let dx=0;dx<obs.grid[0].length;dx++) {
      const gy=obs.vy+dy,gx=obs.vx+dx;
      if (gy<0||gy>=H||gx<0||gx>=W) continue;
      const keys=cf(inits[obs.si],gy,gx); if (!keys) continue;
      const k=keys.d0,fc=t2c(obs.grid[dy][dx]);
      if (!vpD0[k]) vpD0[k]={n:0,counts:new Float64Array(C)}; vpD0[k].n++; vpD0[k].counts[fc]++; } }
  for (const [k,vm] of Object.entries(vpD0)) {
    const bm=model[k];
    if (bm) {
      const pa=bm.a.map(p=>p*CW),post=pa.map((a,c)=>a+vm.counts[c]);
      const tot=post.reduce((a,b)=>a+b,0);
      model[k]={n:bm.n+vm.n,a:post.map(v=>v/tot)};
    } else {
      const tot=vm.n+C*0.1;
      model[k]={n:vm.n,a:Array.from(vm.counts).map(v=>(v+0.1)/tot)};
    }
  }
}

// Per-cell models
function buildCellModels(vpObs, inits) {
  const cm={}; for (let si=0;si<SEEDS;si++) cm[si]={};
  for (const obs of vpObs) {
    for (let dy=0;dy<obs.grid.length;dy++) for (let dx=0;dx<obs.grid[0].length;dx++) {
      const gy=obs.vy+dy,gx=obs.vx+dx;
      if (gy<0||gy>=H||gx<0||gx>=W) continue;
      if (inits[obs.si][gy][gx]===10||inits[obs.si][gy][gx]===5) continue;
      const k=`${gy},${gx}`,fc=t2c(obs.grid[dy][dx]);
      if (!cm[obs.si][k]) cm[obs.si][k]={n:0,counts:new Float64Array(C)};
      cm[obs.si][k].n++; cm[obs.si][k].counts[fc]++; } }
  return cm;
}

function applyPerCell(pred, cellModel, initGrid, pw) {
  for (const [key,cell] of Object.entries(cellModel)) {
    const [y,x]=key.split(',').map(Number);
    if (initGrid[y][x]===10||initGrid[y][x]===5) continue;
    const posterior=new Array(C); let total=0;
    for (let c=0;c<C;c++) { posterior[c]=pred[y][x][c]*pw+cell.counts[c]; total+=posterior[c]; }
    if (total>0) { for (let c=0;c<C;c++) posterior[c]/=total; pred[y][x]=posterior; }
  }
}

// Predict
function predict(grid, model, temp, floor) {
  const pred=[];
  for (let y=0;y<H;y++) { pred[y]=[];
    for (let x=0;x<W;x++) {
      const t=grid[y][x];
      if (t===10) { pred[y][x]=[1,0,0,0,0,0]; continue; }
      if (t===5) { pred[y][x]=[0,0,0,0,0,1]; continue; }
      const keys=cf(grid,y,x);
      if (!keys) { pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      const levels=['d0','d1','d2','d3','d4'],ws=[1.0,0.3,0.15,0.08,0.02];
      const p=[0,0,0,0,0,0]; let wS=0;
      for (let li=0;li<levels.length;li++) {
        const d=model[keys[levels[li]]];
        if (d&&d.n>=1) { const w=ws[li]*Math.pow(d.n,0.5);
          for (let c=0;c<C;c++) p[c]+=w*d.a[c]; wS+=w; } }
      if (wS===0) { pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6]; continue; }
      let s=0;
      for (let c=0;c<C;c++) { p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/temp); s+=p[c]; }
      let s2=0;
      for (let c=0;c<C;c++) { p[c]=Math.max(p[c]/s,floor); s2+=p[c]; }
      for (let c=0;c<C;c++) p[c]/=s2;
      pred[y][x]=p;
    } }
  return pred;
}

// Score prediction against GT
function score(pred, gt) {
  let wklNum=0,wklDen=0;
  for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
    const g=gt[y][x]; let ent=0;
    for (let c=0;c<C;c++) if (g[c]>1e-6) ent-=g[c]*Math.log(g[c]);
    if (ent<0.01) continue;
    let kl=0;
    for (let c=0;c<C;c++) if (g[c]>1e-6) kl+=g[c]*Math.log(g[c]/Math.max(pred[y][x][c],1e-15));
    wklNum+=Math.max(0,kl)*ent; wklDen+=ent;
  }
  const wkl=wklDen>0?wklNum/wklDen:0;
  return Math.max(0,Math.min(100,100*Math.exp(-3*wkl)));
}

// === BENCHMARK ===
log('\n=== BENCHMARKING CONFIGS ON R8, R9, R10 (with VP data + GT) ===\n');

const configs = [
  // Original autopilot params
  { name: 'autopilot_orig', CW: 20, pw: 'autopilot', temp: 1.1, floor: 0.00005, roundW: false },
  // Resubmit script params
  { name: 'resubmit_v1', CW: 6, pw: 'resubmit', temp: 1.0, floor: 0.0001, roundW: true },
  // Sweep CW
  { name: 'CW=2', CW: 2, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'CW=4', CW: 4, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'CW=8', CW: 8, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'CW=12', CW: 12, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'CW=20', CW: 20, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'CW=30', CW: 30, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  // Sweep pw
  { name: 'pw=1', CW: 8, pw: 1, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'pw=5', CW: 8, pw: 5, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'pw=10', CW: 8, pw: 10, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'pw=20', CW: 8, pw: 20, temp: 1.0, floor: 0.0001, roundW: false },
  // Sweep temp
  { name: 'temp=0.9', CW: 8, pw: 3, temp: 0.9, floor: 0.0001, roundW: false },
  { name: 'temp=1.0', CW: 8, pw: 3, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'temp=1.1', CW: 8, pw: 3, temp: 1.1, floor: 0.0001, roundW: false },
  { name: 'temp=1.2', CW: 8, pw: 3, temp: 1.2, floor: 0.0001, roundW: false },
  // No VP (model only)
  { name: 'no_VP', CW: null, pw: null, temp: 1.0, floor: 0.0001, roundW: false },
  { name: 'no_VP_rw', CW: null, pw: null, temp: 1.0, floor: 0.0001, roundW: true },
  // Floor sweep
  { name: 'floor=0.001', CW: 8, pw: 3, temp: 1.0, floor: 0.001, roundW: false },
  { name: 'floor=0.01', CW: 8, pw: 3, temp: 1.0, floor: 0.01, roundW: false },
];

const results = [];
for (const cfg of configs) {
  const roundScores = {};
  for (const testRn of ['R8', 'R9', 'R10']) {
    if (!G[testRn] || !I[testRn]) continue;

    // Build model excluding test round
    const model = buildModel(testRn, cfg.roundW);

    // Fuse VP if applicable
    if (cfg.CW !== null && VP_ROUNDS[testRn]) {
      fuseVP(model, VP_ROUNDS[testRn], I[testRn], cfg.CW);
    }

    // Per-cell models
    const cellModels = (cfg.pw !== null && VP_ROUNDS[testRn])
      ? buildCellModels(VP_ROUNDS[testRn], I[testRn]) : null;

    let roundTotal = 0;
    for (let si = 0; si < SEEDS; si++) {
      if (!I[testRn][si] || !G[testRn][si]) continue;
      let pred = predict(I[testRn][si], model, cfg.temp, cfg.floor);

      if (cellModels && cellModels[si]) {
        let pw_val;
        if (cfg.pw === 'autopilot') {
          // Original autopilot per-cell logic
          applyPerCell(pred, cellModels[si], I[testRn][si], 7); // avg of 2,4,7,15
        } else if (cfg.pw === 'resubmit') {
          applyPerCell(pred, cellModels[si], I[testRn][si], 3); // avg of 1,3,5,10
        } else {
          applyPerCell(pred, cellModels[si], I[testRn][si], cfg.pw);
        }
      }

      roundTotal += score(pred, G[testRn][si]);
    }
    roundScores[testRn] = roundTotal / SEEDS;
  }

  const avg = Object.values(roundScores).reduce((a,b)=>a+b,0) / Object.keys(roundScores).length;
  results.push({ name: cfg.name, ...roundScores, avg });
}

// Sort by average
results.sort((a, b) => b.avg - a.avg);

log('Results (sorted by avg):');
log('Config               | R8       | R9       | R10      | AVG');
log('---------------------|----------|----------|----------|--------');
for (const r of results) {
  const pad = (s, n) => (s + ' '.repeat(n)).slice(0, n);
  log(`${pad(r.name, 20)} | ${(r.R8||0).toFixed(2).padStart(8)} | ${(r.R9||0).toFixed(2).padStart(8)} | ${(r.R10||0).toFixed(2).padStart(8)} | ${r.avg.toFixed(2).padStart(6)}`);
}

log(`\nBest config: ${results[0].name} (avg=${results[0].avg.toFixed(2)})`);
log(`Autopilot orig: ${results.find(r=>r.name==='autopilot_orig')?.avg.toFixed(2)}`);
log(`Model only (no VP): ${results.find(r=>r.name==='no_VP')?.avg.toFixed(2)}`);
