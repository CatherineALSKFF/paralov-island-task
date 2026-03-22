#!/usr/bin/env node
// FAST BENCHMARK — build model once, sweep VP fusion params only
const fs = require('fs'), path = require('path');
const H=40,W=40,SEEDS=5,C=6,DD=path.join(__dirname,'data');
function t2c(t){return(t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0}
function cf(g,y,x){
  const t=g[y][x];if(t===10||t===5)return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;
    if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return{d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`,d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    d2:`D2_${t}_${sa>0?1:0}_${co}`,d3:`D3_${t}_${co}`,d4:`D4_${t}`};
}

// Load data
const I={},G={};
for(let r=1;r<=11;r++){const rn=`R${r}`;
  const ip=path.join(DD,`inits_${rn}.json`),gp=path.join(DD,`gt_${rn}.json`);
  if(fs.existsSync(ip))I[rn]=JSON.parse(fs.readFileSync(ip));
  if(fs.existsSync(gp))G[rn]=JSON.parse(fs.readFileSync(gp));
}
const VP={
  R8:JSON.parse(fs.readFileSync(path.join(DD,'viewport_c5cdf100.json'))),
  R9:JSON.parse(fs.readFileSync(path.join(DD,'viewport_2a341ace.json'))),
  R10:JSON.parse(fs.readFileSync(path.join(DD,'viewport_75e625c3.json'))),
};

// Build base model for each test round (exclude test, include replays)
function buildBase(excludeRn){
  const R={};
  for(let r=1;r<=11;r++){const rn=`R${r}`;
    if(fs.existsSync(path.join(DD,`replays_${rn}.json`)))R[rn]=JSON.parse(fs.readFileSync(path.join(DD,`replays_${rn}.json`)));}
  const TR=Object.keys(I).filter(k=>G[k]&&k!==excludeRn);
  const model={};
  for(const level of['d0','d1','d2','d3','d4']){
    const m={};
    for(const rn of TR){if(!G[rn]||!I[rn])continue;
      for(let si=0;si<SEEDS;si++){if(!I[rn][si]||!G[rn][si])continue;
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
          const keys=cf(I[rn][si],y,x);if(!keys)continue;const k=keys[level];
          if(!m[k])m[k]={n:0,counts:new Float64Array(C)};
          const p=G[rn][si][y][x];
          for(let c=0;c<C;c++)m[k].counts[c]+=p[c]*20;m[k].n+=20;}}}
    for(const rn of TR){if(!R[rn]||!I[rn])continue;
      for(const rep of R[rn]){const g=I[rn][rep.si];if(!g)continue;
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
          const keys=cf(g,y,x);if(!keys)continue;const k=keys[level];
          const fc=t2c(rep.finalGrid[y][x]);
          if(!m[k])m[k]={n:0,counts:new Float64Array(C)};m[k].n++;m[k].counts[fc]++;}}}
    for(const k of Object.keys(m)){
      const tot=Array.from(m[k].counts).reduce((a,b)=>a+b,0)+C*0.05;
      m[k].a=Array.from(m[k].counts).map(v=>(v+0.05)/tot);}
    for(const[k,v]of Object.entries(m)){if(!model[k])model[k]=v;}
  }
  return model;
}

function cloneModel(m){const c={};for(const[k,v]of Object.entries(m))c[k]={n:v.n,a:[...v.a]};return c;}

function fuseVP(model,vpObs,inits,CW){
  const vpD0={};
  for(const obs of vpObs){
    for(let dy=0;dy<obs.grid.length;dy++)for(let dx=0;dx<obs.grid[0].length;dx++){
      const gy=obs.vy+dy,gx=obs.vx+dx;
      if(gy<0||gy>=H||gx<0||gx>=W)continue;
      const keys=cf(inits[obs.si],gy,gx);if(!keys)continue;
      const k=keys.d0,fc=t2c(obs.grid[dy][dx]);
      if(!vpD0[k])vpD0[k]={n:0,counts:new Float64Array(C)};vpD0[k].n++;vpD0[k].counts[fc]++;}}
  for(const[k,vm]of Object.entries(vpD0)){
    const bm=model[k];
    if(bm){const pa=bm.a.map(p=>p*CW),post=pa.map((a,c)=>a+vm.counts[c]);
      const tot=post.reduce((a,b)=>a+b,0);
      model[k]={n:bm.n+vm.n,a:post.map(v=>v/tot)};
    }else{const tot=vm.n+C*0.1;model[k]={n:vm.n,a:Array.from(vm.counts).map(v=>(v+0.1)/tot)};}
  }
}

function buildCM(vpObs,inits){
  const cm={};for(let si=0;si<SEEDS;si++)cm[si]={};
  for(const obs of vpObs){
    for(let dy=0;dy<obs.grid.length;dy++)for(let dx=0;dx<obs.grid[0].length;dx++){
      const gy=obs.vy+dy,gx=obs.vx+dx;
      if(gy<0||gy>=H||gx<0||gx>=W)continue;
      if(inits[obs.si][gy][gx]===10||inits[obs.si][gy][gx]===5)continue;
      const k=`${gy},${gx}`,fc=t2c(obs.grid[dy][dx]);
      if(!cm[obs.si][k])cm[obs.si][k]={n:0,counts:new Float64Array(C)};
      cm[obs.si][k].n++;cm[obs.si][k].counts[fc]++;}}
  return cm;
}

function scoreIt(pred,gt){
  let wN=0,wD=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const g=gt[y][x];let ent=0;
    for(let c=0;c<C;c++)if(g[c]>1e-6)ent-=g[c]*Math.log(g[c]);
    if(ent<0.01)continue;let kl=0;
    for(let c=0;c<C;c++)if(g[c]>1e-6)kl+=g[c]*Math.log(g[c]/Math.max(pred[y][x][c],1e-15));
    wN+=Math.max(0,kl)*ent;wD+=ent;}
  return Math.max(0,Math.min(100,100*Math.exp(-3*(wD>0?wN/wD:0))));
}

function runConfig(baseModel,testRn,CW,pw,temp,floor){
  const model=cloneModel(baseModel);
  if(CW!==null)fuseVP(model,VP[testRn],I[testRn],CW);
  const cm=(pw!==null)?buildCM(VP[testRn],I[testRn]):null;
  let total=0;
  for(let si=0;si<SEEDS;si++){
    if(!I[testRn][si]||!G[testRn][si])continue;
    const grid=I[testRn][si],pred=[];
    for(let y=0;y<H;y++){pred[y]=[];
      for(let x=0;x<W;x++){
        const t=grid[y][x];
        if(t===10){pred[y][x]=[1,0,0,0,0,0];continue;}
        if(t===5){pred[y][x]=[0,0,0,0,0,1];continue;}
        const keys=cf(grid,y,x);
        if(!keys){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
        const levels=['d0','d1','d2','d3','d4'],ws=[1,0.3,0.15,0.08,0.02];
        const p=[0,0,0,0,0,0];let wS=0;
        for(let li=0;li<levels.length;li++){const d=model[keys[levels[li]]];
          if(d&&d.n>=1){const w=ws[li]*Math.pow(d.n,0.5);
            for(let c=0;c<C;c++)p[c]+=w*d.a[c];wS+=w;}}
        if(wS===0){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
        let s=0;for(let c=0;c<C;c++){p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/temp);s+=p[c];}
        let s2=0;for(let c=0;c<C;c++){p[c]=Math.max(p[c]/s,floor);s2+=p[c];}
        for(let c=0;c<C;c++)p[c]/=s2;
        pred[y][x]=p;}}
    // Per-cell
    if(cm&&cm[si]){for(const[key,cell]of Object.entries(cm[si])){
      const[y,x]=key.split(',').map(Number);
      if(grid[y][x]===10||grid[y][x]===5)continue;
      const post=new Array(C);let tot=0;
      for(let c=0;c<C;c++){post[c]=pred[y][x][c]*pw+cell.counts[c];tot+=post[c];}
      if(tot>0)for(let c=0;c<C;c++)pred[y][x][c]=post[c]/tot;}}
    total+=scoreIt(pred,G[testRn][si]);
  }
  return total/SEEDS;
}

// BUILD BASE MODELS (once per test round)
console.log('Building base models...');
const bases={};
for(const rn of['R8','R9','R10']){
  console.log(`  ${rn}...`);
  bases[rn]=buildBase(rn);
}
console.log('Done. Sweeping configs...\n');

// SWEEP
const cfgs=[];
// CW sweep
for(const CW of[null,2,4,6,8,10,15,20,30,50])
  for(const pw of[null,1,3,5,10,20])
    for(const temp of[0.9,1.0,1.1])
      for(const floor of[0.0001,0.001]){
        if(CW===null&&pw!==null)continue; // no VP model but per-cell makes no sense
        if(CW!==null&&pw===null)cfgs.push({CW,pw:null,temp,floor}); // VP model only
        if(CW===null&&pw===null)cfgs.push({CW,pw,temp,floor}); // pure model
        if(CW!==null&&pw!==null)cfgs.push({CW,pw,temp,floor}); // VP + per-cell
      }
// Deduplicate
const seen=new Set();const uniq=[];
for(const c of cfgs){const k=JSON.stringify(c);if(!seen.has(k)){seen.add(k);uniq.push(c);}}
console.log(`Testing ${uniq.length} configs...\n`);

const results=[];
for(let i=0;i<uniq.length;i++){
  const c=uniq[i];
  let sum=0,n=0;
  for(const rn of['R8','R9','R10']){
    sum+=runConfig(bases[rn],rn,c.CW,c.pw,c.temp,c.floor);n++;
  }
  results.push({...c,avg:sum/n});
  if((i+1)%50===0)process.stdout.write(`  ${i+1}/${uniq.length}\n`);
}

results.sort((a,b)=>b.avg-a.avg);
console.log('\n=== TOP 20 CONFIGS ===');
console.log('CW\tpw\ttemp\tfloor\t\tAVG');
for(let i=0;i<Math.min(20,results.length);i++){
  const r=results[i];
  console.log(`${r.CW||'none'}\t${r.pw||'none'}\t${r.temp}\t${r.floor}\t\t${r.avg.toFixed(2)}`);
}

// Show autopilot original for comparison
const orig=results.find(r=>r.CW===20&&r.pw===null&&r.temp===1.1&&r.floor===0.0001);
const noVP=results.find(r=>r.CW===null&&r.pw===null&&r.temp===1.0&&r.floor===0.0001);
console.log(`\nAutopilot (CW=20,no-pc,t=1.1): ${orig?orig.avg.toFixed(2):'N/A'}`);
console.log(`No VP (model only): ${noVP?noVP.avg.toFixed(2):'N/A'}`);
console.log(`BEST: CW=${results[0].CW} pw=${results[0].pw} temp=${results[0].temp} floor=${results[0].floor} → ${results[0].avg.toFixed(2)}`);
