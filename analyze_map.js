#!/usr/bin/env node
const https = require('https');
const BASE = 'https://api.ainm.no/astar-island';
const H = 40, W = 40, TOKEN = process.argv[2] || '';

function api(method, path) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + path);
    const opts = { hostname: url.hostname, path: url.pathname, method,
      headers: { 'Authorization': 'Bearer ' + TOKEN }};
    const req = https.request(opts, res => {
      let data = ''; res.on('data', c => data += c);
      res.on('end', () => resolve(JSON.parse(data)));
    }); req.on('error', reject); req.end();
  });
}

function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
    if(dy===0 && dx===0) continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W) continue;
    const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;
    if(nt===10)co=1;
    if(nt===4)fN++;
  }
  for(let dy=-2;dy<=2;dy++) for(let dx=-2;dx<=2;dx++) {
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1) continue;
    const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W) continue;
    if(g[ny][nx]===1||g[ny][nx]===2) sR2++;
  }
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return [`D0_${t}_${sa}_${co}_${sb2}_${fb}`,`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,`D2_${t}_${sa>0?1:0}_${co}`,`D3_${t}_${co}`,`D4_${t}`];
}

async function main() {
  const roundId = process.argv[3] || 'ae78003a-4efe-425a-881a-d16a39bca0ad';
  const data = await api('GET', '/rounds/' + roundId);

  for (let si = 0; si < 1; si++) {
    const grid = data.initial_states[si].grid;

    let ocean=0, mountain=0, forest=0, settlement=0, port=0, plains=0;
    for (let y=0; y<H; y++) for (let x=0; x<W; x++) {
      const t = grid[y][x];
      if (t===10) ocean++;
      else if (t===5) mountain++;
      else if (t===4) forest++;
      else if (t===1) settlement++;
      else if (t===2) port++;
      else if (t===11||t===0) plains++;
    }
    console.log(`Seed ${si} map composition:`);
    console.log(`  Ocean: ${ocean}  Mountain: ${mountain}  Forest: ${forest}`);
    console.log(`  Settlement: ${settlement}  Port: ${port}  Plains: ${plains}`);
    console.log(`  Total dynamic: ${H*W - ocean - mountain}`);

    const d0keys = {};
    for (let y=0; y<H; y++) for (let x=0; x<W; x++) {
      const keys = cf(grid, y, x);
      if (keys === null) continue;
      d0keys[keys[0]] = (d0keys[keys[0]] || 0) + 1;
    }

    const sorted = Object.entries(d0keys).sort((a,b)=>b[1]-a[1]);
    console.log(`\nD0 feature keys: ${Object.keys(d0keys).length}`);
    console.log('Top 15 most common:');
    for (const [k,v] of sorted.slice(0,15)) console.log(`  ${k}: ${v} cells`);

    // Simulate viewport coverage
    const starts = [0, 13, 25];
    let dynPerPass = 0;
    for (const vy of starts) for (const vx of starts) {
      for (let dy=0; dy<15; dy++) for (let dx=0; dx<15; dx++) {
        const gy = vy+dy, gx = vx+dx;
        if (gy>=H || gx>=W) continue;
        if (grid[gy][gx] !== 10 && grid[gy][gx] !== 5) dynPerPass++;
      }
    }
    console.log(`\nDynamic cells per pass (9 viewports): ${dynPerPass}`);
    console.log(`With 5 passes: ${dynPerPass*5} total observations`);
    console.log(`Avg obs per D0 key: ${(dynPerPass*5/Object.keys(d0keys).length).toFixed(1)}`);
    console.log(`+ 5 extra viewports: ~${Math.round(dynPerPass*5/9*14/Object.keys(d0keys).length)} more per key`);
  }
}

main().catch(console.error);
