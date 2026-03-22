const { H, W, selectClosestRounds } = require('./shared');

function mergeBkts(p,r){const m={};for(const n of r){const b=p[String(n)];if(!b)continue;for(const[k,v]of Object.entries(b)){if(!m[k])m[k]={count:0,sum:[0,0,0,0,0,0]};m[k].count+=v.count;for(let c=0;c<6;c++)m[k].sum[c]+=v.sum[c];}}const o={};for(const[k,v]of Object.entries(m))o[k]=v.sum.map(s=>s/v.count);return o;}

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
  const K = cfg.K || 3;
  const baseK1w = cfg.K1W !== undefined ? cfg.K1W : 0.15;
  const varScale = cfg.VAR_SCALE !== undefined ? cfg.VAR_SCALE : 3.0;

  const tg = gr[String(tr)] || 0.15;
  const cand = {...gr}; delete cand[String(tr)];
  const closeK = selectClosestRounds(cand, tg, K);
  const close1 = [closeK[0]];
  const allR = Object.keys(prb).map(Number).filter(n=>n!==tr);

  const m1 = mergeBkts(prb, close1);
  const mK = mergeBkts(prb, closeK);
  const mAll = mergeBkts(prb, allR);

  // Compute per-key cross-round variance for closeK rounds
  const keyVar = {};
  for (const key of new Set([...Object.keys(mK)])) {
    let totalVar = 0;
    let nRounds = 0;
    for (const rn of closeK) {
      const b = prb[String(rn)];
      if (!b || !b[key]) continue;
      const rd = b[key].sum.map(s => s / b[key].count);
      for (let c = 0; c < 6; c++) {
        const diff = rd[c] - mK[key][c];
        totalVar += diff * diff;
      }
      nRounds++;
    }
    keyVar[key] = nRounds > 1 ? totalVar / nRounds : 0.1;
  }

  const sp = new Set(), sl = [];
  for (const s of setts) { sp.add(s.y*W+s.x); sl.push(s); }

  const pred = [];
  for (let y=0;y<H;y++) { const row = [];
    for (let x=0;x<W;x++) {
      const {std, rich} = computeKeys(grid, sp, sl, y, x);

      // Lookup
      const useKey = (rich !== std && (mK[rich] || mAll[rich])) ? rich : std;
      const p1 = m1[useKey] || m1[std] || mAll[useKey] || mAll[std] || (() => {
        const fb=std.slice(0,-1); return m1[fb]||mK[fb]||mAll[fb];
      })() || [1/6,1/6,1/6,1/6,1/6,1/6];
      const pK = mK[useKey] || mK[std] || mAll[useKey] || mAll[std] || (() => {
        const fb=std.slice(0,-1); return mK[fb]||mAll[fb];
      })() || [1/6,1/6,1/6,1/6,1/6,1/6];

      // Adaptive K1 weight: higher when variance is high
      const v = keyVar[useKey] || keyVar[std] || 0.05;
      const k1w = Math.min(0.5, baseK1w + varScale * v);

      const dist = new Array(6);
      for (let c=0;c<6;c++) {
        dist[c] = Math.max(k1w * p1[c] + (1-k1w) * pK[c], floor);
      }
      const sum = dist.reduce((a,b)=>a+b,0);
      row.push(dist.map(v=>v/sum));
    }
    pred.push(row);
  }
  return pred;
}
module.exports = { predict };
