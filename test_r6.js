#!/usr/bin/env node
// Test model on R6 (our worst round) to validate improvements
const fs = require('fs');
const path = require('path');
const DATA_DIR = path.join(__dirname, 'data');
const H = 40, W = 40, SEEDS = 5, C = 6;

function t2c(t) { return (t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0; }
function cf(g, y, x) {
  const t = g[y][x]; if (t === 10 || t === 5) return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(dy===0&&dx===0)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;
  }
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;
  }
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return { d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`, d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`, d2:`D2_${t}_${sa>0?1:0}_${co}`, d3:`D3_${t}_${co}`, d4:`D4_${t}` };
}

function buildModel(replaysMap, initsMap, roundNames, level, alpha) {
  const m = {};
  for (const rn of roundNames) {
    if (!replaysMap[rn]) continue;
    for (const rep of replaysMap[rn]) {
      const g = initsMap[rn][rep.si]; if (!g) continue;
      for (let y=0;y<H;y++)for(let x=0;x<W;x++){
        const keys=cf(g,y,x);if(!keys)continue;
        const k=keys[level],fc=t2c(rep.finalGrid[y][x]);
        if(!m[k])m[k]={n:0,counts:new Float64Array(C)};
        m[k].n++;m[k].counts[fc]++;
      }
    }
  }
  for(const k of Object.keys(m)){
    const total=m[k].n+C*alpha;
    m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/total);
  }
  return m;
}

function predict(grid, model, fl=0.00005) {
  const pred=[];
  for(let y=0;y<H;y++){pred[y]=[];
    for(let x=0;x<W;x++){
      const t=grid[y][x];
      if(t===10){pred[y][x]=[1,0,0,0,0,0];continue;}
      if(t===5){pred[y][x]=[0,0,0,0,0,1];continue;}
      const keys=cf(grid,y,x);
      if(!keys){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
      const levels=['d0','d1','d2','d3','d4'];
      const ws=[1.0,0.3,0.15,0.08,0.02];
      const p=[0,0,0,0,0,0];let wS=0;
      for(let li=0;li<levels.length;li++){
        const d=model[keys[levels[li]]];
        if(d&&d.n>=1){const w=ws[li]*Math.pow(d.n,0.5);
          for(let c=0;c<C;c++)p[c]+=w*d.a[c];wS+=w;}
      }
      if(wS===0){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
      let s=0;for(let c=0;c<C;c++){p[c]/=wS;if(p[c]<fl)p[c]=fl;s+=p[c];}
      for(let c=0;c<C;c++)p[c]/=s;pred[y][x]=p;
    }
  }
  return pred;
}

function score(pred,gt){
  let tE=0,tWK=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const p=gt[y][x],q=pred[y][x];
    let e=0;for(let c=0;c<C;c++)if(p[c]>0.001)e-=p[c]*Math.log(p[c]);
    if(e<0.01)continue;
    let kl=0;for(let c=0;c<C;c++)if(p[c]>0.001)kl+=p[c]*Math.log(p[c]/Math.max(q[c],1e-10));
    tE+=e;tWK+=e*kl;
  }
  if(tE===0)return 100;
  return Math.max(0,Math.min(100,100*Math.exp(-3*tWK/tE)));
}

// Load data
const replaysMap={},initsMap={};
for(const rn of['R1','R2','R4','R5','R6']){
  const rf=path.join(DATA_DIR,'replays_'+rn+'.json');
  const inf=path.join(DATA_DIR,'inits_'+rn+'.json');
  if(fs.existsSync(rf)&&fs.existsSync(inf)){
    replaysMap[rn]=JSON.parse(fs.readFileSync(rf));
    initsMap[rn]=JSON.parse(fs.readFileSync(inf));
  }
}
const gt6=JSON.parse(fs.readFileSync(path.join(DATA_DIR,'gt_R6.json')));
const inits6=initsMap['R6'];

console.log('=== Held-out R6 Test ===');
console.log('Available replays:', Object.entries(replaysMap).map(([k,v])=>k+'='+v.length).join(', '));

const trainExR6 = Object.keys(replaysMap).filter(r => r !== 'R6');
console.log('\nTraining on:', trainExR6.join(', '), '(held out: R6)');

// Test D0 vs D1 (cross-round only, no viewport)
for (const level of ['d0', 'd1']) {
  const model = buildModel(replaysMap, initsMap, trainExR6, level, 0.05);
  const scores = [];
  for (let si=0;si<SEEDS;si++) scores.push(score(predict(inits6[si], model), gt6[si]));
  console.log(level + ' cross-round: ' + (scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(2) + ' [' + scores.map(s=>s.toFixed(1)).join(', ') + ']');
}

// With simulated viewport (using R6 replays as observations)
console.log('\n=== With simulated viewport (50 replay frames as observations) ===');
const r6Reps = replaysMap['R6'];
if (r6Reps) {
  for (const nVP of [10, 25, 50]) {
    const vpReps = r6Reps.slice(0, nVP);
    for (const cw of [10, 20, 30, 50]) {
      const crossD0 = buildModel(replaysMap, initsMap, trainExR6, 'd0', 0.05);
      // Build viewport model from R6 replays
      const vpModel = {};
      for (const rep of vpReps) {
        for (let y=0;y<H;y++) for(let x=0;x<W;x++){
          const keys=cf(inits6[0],y,x);if(!keys)continue;
          const k=keys.d0,fc=t2c(rep.finalGrid[y][x]);
          if(!vpModel[k])vpModel[k]={n:0,counts:new Float64Array(C)};
          vpModel[k].n++;vpModel[k].counts[fc]++;
        }
      }
      for(const k of Object.keys(vpModel)){
        const total=vpModel[k].n+C*0.1;
        vpModel[k].a=Array.from(vpModel[k].counts).map(v=>(v+0.1)/total);
      }
      // Fuse
      const fused={};
      const allK=new Set([...Object.keys(crossD0),...Object.keys(vpModel)]);
      for(const k of allK){
        const cm=crossD0[k],vm=vpModel[k];
        if(cm&&vm){
          const prior=cm.a.map(p=>p*cw);
          const post=prior.map((a,c)=>a+vm.counts[c]);
          const total=post.reduce((a,b)=>a+b,0);
          fused[k]={n:cm.n+vm.n,a:post.map(v=>v/total)};
        }else if(vm){fused[k]={n:vm.n,a:vm.a.slice()};}
        else{fused[k]={n:cm.n,a:cm.a.slice()};}
      }
      // Add fallbacks
      for(const lvl of['d1','d2','d3','d4']){
        const cross=buildModel(replaysMap,initsMap,trainExR6,lvl,0.05);
        for(const[k,v]of Object.entries(cross)){if(!fused[k])fused[k]=v;}
      }
      const scores=[];
      for(let si=0;si<SEEDS;si++) scores.push(score(predict(inits6[si],fused),gt6[si]));
      console.log(`n=${nVP} cw=${cw}: ${(scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(2)} [${scores.map(s=>s.toFixed(1)).join(', ')}]`);
    }
  }

  // Test pure viewport model (no cross-round)
  console.log('\n=== Pure viewport (no cross-round prior) ===');
  for (const nVP of [50, 100, 200, 500]) {
    const vpReps = r6Reps.slice(0, Math.min(nVP, r6Reps.length));
    const vpModel = {};
    for (const rep of vpReps) {
      for (let y=0;y<H;y++) for(let x=0;x<W;x++){
        const keys=cf(inits6[0],y,x);if(!keys)continue;
        const k=keys.d0,fc=t2c(rep.finalGrid[y][x]);
        if(!vpModel[k])vpModel[k]={n:0,counts:new Float64Array(C)};
        vpModel[k].n++;vpModel[k].counts[fc]++;
      }
    }
    for(const k of Object.keys(vpModel)){
      const total=vpModel[k].n+C*0.1;
      vpModel[k].a=Array.from(vpModel[k].counts).map(v=>(v+0.1)/total);
    }
    const scores=[];
    for(let si=0;si<SEEDS;si++) scores.push(score(predict(inits6[si],vpModel),gt6[si]));
    console.log(`n=${nVP}: ${(scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(2)} [${scores.map(s=>s.toFixed(1)).join(', ')}]`);
  }
}

console.log('\nActual R6 score: 69.07 (no viewport data - all queries wasted in Chrome crashes)');
console.log('R6 was the hardest round: 1394 high-entropy cells, avg entropy 0.906');
