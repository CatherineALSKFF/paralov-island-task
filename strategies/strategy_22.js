const { H, W, selectClosestRounds } = require('./shared');

function computeKeys(grid, sp, sl, y, x) {
  const v = grid[y][x];
  if (v === 10) return { std: 'O', rich: 'O' };
  if (v === 5) return { std: 'M', rich: 'M' };
  const t = v === 4 ? 'F' : (v === 1 || v === 2) ? 'S' : 'P';
  let nS = 0;
  for (let dy=-3;dy<=3;dy++) for (let dx=-3;dx<=3;dx++) {
    if (!dy&&!dx) continue; const ny=y+dy,nx=x+dx;
    if (ny>=0&&ny<H&&nx>=0&&nx<W&&sp.has(ny*W+nx)) nS++;
  }
  let coast = false;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny=y+dy,nx=x+dx;
    if (ny>=0&&ny<H&&nx>=0&&nx<W&&grid[ny][nx]===10) coast=true;
  }
  const std = t+(nS===0?'0':nS<=2?'1':nS<=5?'2':'3')+(coast?'c':'');
  let rich = std;
  if (nS===0&&t!=='S'&&sl.length>0) {
    let md=999;
    for (const s of sl) md=Math.min(md,Math.max(Math.abs(s.y-y),Math.abs(s.x-x)));
    rich=std+(md<=4?'n':md<=8?'m':'f');
  } else if (nS>=1&&nS<=2&&t!=='S') {
    let md=999;
    for (const s of sl) { const d=Math.max(Math.abs(s.y-y),Math.abs(s.x-x)); if(d<=3) md=Math.min(md,d); }
    rich=std+(md<=1?'a':md<=2?'b':'');
  }
  return { std, rich };
}

function predict(grid, setts, prb, gr, tr, cfg) {
  const floor = cfg.FLOOR || 0.0001;
  const bw = cfg.BW || 0.03;
  const minW = cfg.MIN_W || 0.005;

  const tg = gr[String(tr)] || 0.15;

  // Compute per-round weights
  const weights = {};
  for (const [rn, rate] of Object.entries(gr)) {
    if (parseInt(rn) === tr) continue;
    if (!prb[rn]) continue;
    const dist = Math.abs(rate - tg);
    weights[rn] = Math.max(minW, Math.exp(-0.5 * (dist / bw) ** 2));
  }

  // Build weighted model per feature key
  const model = {};
  for (const [rn, w] of Object.entries(weights)) {
    const b = prb[rn]; if (!b) continue;
    for (const [key, v] of Object.entries(b)) {
      if (!model[key]) model[key] = { dist: [0,0,0,0,0,0], totalW: 0 };
      const rd = v.sum.map(s => s / v.count);
      for (let c = 0; c < 6; c++) model[key].dist[c] += w * rd[c];
      model[key].totalW += w;
    }
  }
  // Normalize
  const modelDist = {};
  for (const [k, v] of Object.entries(model)) {
    modelDist[k] = v.dist.map(s => s / v.totalW);
  }

  const sp = new Set(), sl = [];
  for (const s of setts) { sp.add(s.y*W+s.x); sl.push(s); }

  const pred = [];
  for (let y=0;y<H;y++) { const row = [];
    for (let x=0;x<W;x++) {
      const {std, rich} = computeKeys(grid, sp, sl, y, x);

      let p = null;
      if (rich !== std) p = modelDist[rich];
      if (!p) p = modelDist[std];
      if (!p) { const fb=std.slice(0,-1); p=modelDist[fb]; }
      if (!p) p = [1/6,1/6,1/6,1/6,1/6,1/6];

      const dist = p.map(v => Math.max(v, floor));
      const sum = dist.reduce((a,b)=>a+b,0);
      row.push(dist.map(v=>v/sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
