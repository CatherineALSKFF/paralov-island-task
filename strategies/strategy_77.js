const { H, W, getFeatureKey } = require('./shared');

function predict(initGrid, settlements, perRoundBuckets, growthRates, testRound, config) {
  const floor = config.FLOOR || 0.0001;
  const sigma = config.sigma || 0.060;
  const targetGrowth = growthRates[String(testRound)] || 0.15;

  const settPos = new Set();
  for (const s of settlements) settPos.add(s.y * W + s.x);

  // Precompute nearest settlement distance (Chebyshev)
  const nearestDist = new Float32Array(H * W).fill(99);
  for (const s of settlements) {
    for (let y = Math.max(0, s.y - 12); y <= Math.min(H - 1, s.y + 12); y++)
      for (let x = Math.max(0, s.x - 12); x <= Math.min(W - 1, s.x + 12); x++) {
        const d = Math.max(Math.abs(s.y - y), Math.abs(s.x - x));
        const idx = y * W + x;
        if (d < nearestDist[idx]) nearestDist[idx] = d;
      }
  }

  // Enriched feature key: adds distance-based sub-categories
  function getEK(y, x) {
    const base = getFeatureKey(initGrid, settPos, y, x);
    if (base === 'O' || base === 'M' || base[0] === 'S') return base;
    const b = base[1], md = nearestDist[y * W + x];
    if (b === '0') return base + (md === 4 ? 'n' : md <= 8 ? 'm' : 'f');
    if (b === '1') { if (md === 1) return base + 'a'; if (md === 2) return base + 'b'; }
    return base;
  }

  const allRounds = Object.keys(perRoundBuckets).map(Number).filter(n => n !== testRound);

  // Gaussian round weights
  const rwt = {};
  let twt = 0;
  for (const rn of allRounds) {
    const d = (growthRates[String(rn)] || 0.15) - targetGrowth;
    rwt[rn] = Math.exp(-d * d / (2 * sigma * sigma));
    twt += rwt[rn];
  }
  for (const rn of allRounds) rwt[rn] /= twt;

  // Count-weighted Gaussian merge
  const gm = {}, gc = {};
  for (const rn of allRounds) {
    const b = perRoundBuckets[String(rn)];
    if (!b) continue;
    const w = rwt[rn];
    for (const [key, val] of Object.entries(b)) {
      if (!gm[key]) { gm[key] = [0,0,0,0,0,0]; gc[key] = 0; }
      gc[key] += w * val.count;
      for (let c = 0; c < 6; c++) gm[key][c] += w * val.sum[c];
    }
  }
  const mp = {};
  for (const [k, v] of Object.entries(gm)) if (gc[k] > 0) mp[k] = v.map(s => s / gc[k]);

  // Residual 'd' keys for dist=3 cells
  for (const [base, s1, s2] of [['P1','P1a','P1b'],['P1c','P1ca','P1cb'],['F1','F1a','F1b'],['F1c','F1ca','F1cb']]) {
    if (!gc[base]) continue;
    const r = gc[base] - (gc[s1] || 0) - (gc[s2] || 0);
    if (r < 3) continue;
    const d = [0,0,0,0,0,0];
    for (let c = 0; c < 6; c++) d[c] = Math.max(0, (gm[base][c] - (gm[s1]?gm[s1][c]:0) - (gm[s2]?gm[s2][c]:0)) / r);
    const ds = d.reduce((a,b) => a+b, 0);
    if (ds > 0) mp[base + 'd'] = d.map(v => v/ds);
  }

  // Adaptive-bandwidth LOESS local linear regression
  const lc = {};
  function loess(key) {
    if (lc[key] !== undefined) return lc[key];
    const pts = [];
    for (const rn of allRounds) {
      const b = perRoundBuckets[String(rn)];
      if (!b || !b[key] || b[key].count < 3) continue;
      pts.push({ g: growthRates[String(rn)] || 0.15, avg: b[key].sum.map(v => v / b[key].count), n: b[key].count });
    }
    if (pts.length < 4) { lc[key] = null; return null; }

    const dists = pts.map(pt => Math.abs(pt.g - targetGrowth)).sort((a, b) => a - b);
    const bw = Math.max(0.04, Math.min(0.20, dists[Math.min(3, dists.length - 1)] * 1.5));

    const r = [0,0,0,0,0,0];
    for (let c = 0; c < 6; c++) {
      let sw=0, swg=0, swp=0, swgg=0, swgp=0;
      for (const pt of pts) {
        const gd = pt.g - targetGrowth;
        const w = Math.exp(-gd*gd/(2*bw*bw)) * Math.sqrt(pt.n);
        sw += w; swg += w*pt.g; swp += w*pt.avg[c]; swgg += w*pt.g*pt.g; swgp += w*pt.g*pt.avg[c];
      }
      const dn = sw*swgg - swg*swg;
      if (Math.abs(dn) < 1e-12) r[c] = swp/sw;
      else {
        const slope = (sw*swgp - swg*swp) / dn;
        const intercept = (swp - slope*swg) / sw;
        r[c] = Math.max(0, intercept + slope * targetGrowth);
      }
    }
    const s = r.reduce((a,b) => a+b, 0);
    lc[key] = s > 0 ? r.map(v => v/s) : null;
    return lc[key];
  }

  // Per-key disagreement (weighted L2 deviation across rounds)
  const dc = {};
  function getDis(key) {
    if (dc[key] !== undefined) return dc[key];
    const avg = mp[key];
    if (!avg) { dc[key] = 0; return 0; }
    let dis = 0, dtw = 0;
    for (const rn of allRounds) {
      const b = perRoundBuckets[String(rn)];
      if (!b || !b[key] || b[key].count < 3) continue;
      const rd = b[key].sum.map(v => v / b[key].count);
      let d = 0;
      for (let c = 0; c < 6; c++) { const dd = rd[c] - avg[c]; d += dd*dd; }
      dis += rwt[rn] * Math.sqrt(d);
      dtw += rwt[rn];
    }
    dc[key] = dtw > 0 ? dis/dtw : 0;
    return dc[key];
  }

  function lkup(rk, bk) {
    if (mp[rk]) return { p: mp[rk], k: rk };
    if (mp[bk]) return { p: mp[bk], k: bk };
    for (let l = bk.length-1; l >= 1; l--) { const fb = bk.slice(0,l); if (mp[fb]) return { p: mp[fb], k: fb }; }
    return { p: [1/6,1/6,1/6,1/6,1/6,1/6], k: '' };
  }

  // Phase 1: raw predictions
  const rawPred = [];
  const isStatic = [];
  for (let y = 0; y < H; y++) {
    rawPred.push(new Array(W));
    isStatic.push(new Array(W));
    for (let x = 0; x < W; x++) {
      const bk = getFeatureKey(initGrid, settPos, y, x);
      if (bk === 'O' || bk === 'M') {
        isStatic[y][x] = true;
        rawPred[y][x] = bk === 'O' ? [1,0,0,0,0,0] : [0,0,0,0,0,1];
        continue;
      }
      isStatic[y][x] = false;
      const rk = getEK(y, x);

      const { p: gp, k: uk } = lkup(rk, bk);
      const lp = loess(uk) || loess(bk);
      const dis = getDis(uk) || getDis(bk);

      // Blend LOESS + Gaussian: reduce LOESS when disagreement is high
      let prior;
      if (lp && gp) {
        const lw = Math.max(0.15, 1.0 - dis * 1.5);
        prior = gp.map((g, c) => (1-lw)*g + lw*lp[c]);
      } else if (lp) {
        prior = lp;
      } else {
        prior = [...gp];
      }

      // Mild regularization toward coarser key
      if (uk.length > 1) {
        const ck = uk.length > 2 ? uk.slice(0,-1) : uk[0];
        if (mp[ck]) for (let c=0;c<6;c++) prior[c] = 0.95*prior[c] + 0.05*mp[ck][c];
      }

      // Temperature scaling for high disagreement
      if (dis > 0.12) {
        const temp = 1.0 + 0.6 * Math.min(dis, 0.8);
        let s = 0;
        for (let c=0;c<6;c++) { prior[c] = Math.pow(Math.max(prior[c], 1e-12), 1/temp); s += prior[c]; }
        for (let c=0;c<6;c++) prior[c] /= s;
      }

      rawPred[y][x] = prior;
    }
  }

  // Phase 2: mild spatial smoothing + adaptive floor
  const pred = [];
  const spatialW = 0.04;
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      if (isStatic[y][x]) { row.push(rawPred[y][x]); continue; }
      const raw = rawPred[y][x];

      const nAvg = [0,0,0,0,0,0];
      let nw = 0;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        if (dy === 0 && dx === 0) continue;
        const ny = y+dy, nx = x+dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W || isStatic[ny][nx]) continue;
        const d = Math.sqrt(dy*dy + dx*dx);
        const w = 1/(1+d);
        for (let c = 0; c < 6; c++) nAvg[c] += w * rawPred[ny][nx][c];
        nw += w;
      }

      let smoothed;
      if (nw > 0) {
        smoothed = raw.map((v, c) => (1-spatialW)*v + spatialW*(nAvg[c]/nw));
      } else {
        smoothed = raw;
      }

      const rk = getEK(y, x);
      const bk = getFeatureKey(initGrid, settPos, y, x);
      const dis = getDis(rk) || getDis(bk);
      const aFloor = floor + 0.002 * Math.min(dis, 0.8);

      const fl = smoothed.map(v => Math.max(v, aFloor));
      const s = fl.reduce((a,b) => a+b, 0);
      row.push(fl.map(v => v/s));
    }
    pred.push(row);
  }
  return pred;
}

module.exports = { predict };
