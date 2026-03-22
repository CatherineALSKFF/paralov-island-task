#!/usr/bin/env node
// Build R12 predictions using BEST benchmarked config: CW=50, temp=0.9, no per-cell
// Saves predictions to data/r12_preds.json for browser submission
const fs=require('fs'),path=require('path');
const H=40,W=40,SEEDS=5,C=6,DD=path.join(__dirname,'data');
function t2c(t){return(t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0;}
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

// Load all training data
const I={},G={},R={};
for(let r=1;r<=11;r++){const rn=`R${r}`;
  if(fs.existsSync(path.join(DD,`inits_${rn}.json`)))I[rn]=JSON.parse(fs.readFileSync(path.join(DD,`inits_${rn}.json`)));
  if(fs.existsSync(path.join(DD,`gt_${rn}.json`)))G[rn]=JSON.parse(fs.readFileSync(path.join(DD,`gt_${rn}.json`)));
  if(fs.existsSync(path.join(DD,`replays_${rn}.json`)))R[rn]=JSON.parse(fs.readFileSync(path.join(DD,`replays_${rn}.json`)));
}
const TR=Object.keys(I).filter(k=>G[k]);
console.log(`Training: ${TR.join(', ')}`);
for(const rn of TR)console.log(`  ${rn}: GT=yes replays=${R[rn]?R[rn].length:0}`);

// Build model (GT + replays, same as fast_bench)
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
console.log(`Model: ${Object.keys(model).length} keys`);

// Load VP data for R12
const CW=50,TEMP=0.9,FLOOR=0.0001;
const vpFile=path.join(DD,'viewport_795bfb1f.json');
const vpObs=fs.existsSync(vpFile)?JSON.parse(fs.readFileSync(vpFile)):[];
console.log(`VP: ${vpObs.length} observations`);

// Load R12 inits
const inits=JSON.parse(fs.readFileSync(path.join(DD,'inits_R12.json')));

// VP fusion
if(vpObs.length>0){
  const vpD0={};
  for(const obs of vpObs){
    for(let dy=0;dy<obs.grid.length;dy++)for(let dx=0;dx<obs.grid[0].length;dx++){
      const gy=obs.vy+dy,gx=obs.vx+dx;
      if(gy<0||gy>=H||gx<0||gx>=W)continue;
      const keys=cf(inits[obs.si],gy,gx);if(!keys)continue;
      const k=keys.d0,fc=t2c(obs.grid[dy][dx]);
      if(!vpD0[k])vpD0[k]={n:0,counts:new Float64Array(C)};vpD0[k].n++;vpD0[k].counts[fc]++;}}
  let fused=0;
  for(const[k,vm]of Object.entries(vpD0)){
    if(model[k]){
      const pa=model[k].a.map(p=>p*CW),post=pa.map((a,c)=>a+vm.counts[c]);
      const tot=post.reduce((a,b)=>a+b,0);
      model[k]={n:model[k].n+vm.n,a:post.map(v=>v/tot)};fused++;}
  }
  console.log(`VP fused: ${fused} D0 keys`);
}

// Predict all 5 seeds
const preds=[];
for(let si=0;si<SEEDS;si++){
  const grid=inits[si],pred=[];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
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
    let s=0;for(let c=0;c<C;c++){p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/TEMP);s+=p[c];}
    let s2=0;for(let c=0;c<C;c++){p[c]=Math.max(p[c]/s,FLOOR);s2+=p[c];}
    for(let c=0;c<C;c++)p[c]/=s2;
    pred[y][x]=p;
  }}
  // Validate
  let valid=true;
  for(let y=0;y<H&&valid;y++)for(let x=0;x<W&&valid;x++){
    const s=pred[y][x].reduce((a,b)=>a+b,0);
    if(Math.abs(s-1)>0.02||pred[y][x].some(v=>v<0))valid=false;}
  console.log(`Seed ${si}: valid=${valid}`);
  preds.push(pred);
}

// Save to file
const outFile=path.join(DD,'r12_preds_best.json');
fs.writeFileSync(outFile,JSON.stringify(preds));
console.log(`\nSaved to ${outFile} (${(fs.statSync(outFile).size/1024/1024).toFixed(1)}MB)`);
console.log(`Config: CW=${CW} temp=${TEMP} floor=${FLOOR} VP=${vpObs.length} replays=yes`);
console.log('Load in browser and submit with credentials:include');
