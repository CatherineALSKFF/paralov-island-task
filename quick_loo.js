#!/usr/bin/env node
// Quick LOO test — preloads data once, tests fast
'use strict';
const fs=require('fs'),path=require('path');
const H=40,W=40,SEEDS=5,C=6,DD=path.join(__dirname,'data');
function t2c(t){return(t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0;}
function fk(g,y,x){
  const t=g[y][x];if(t===10||t===5)return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return{d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`,d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    d2:`D2_${t}_${sa>0?1:0}_${co}`,d3:`D3_${t}_${co}`,d4:`D4_${t}`};
}
function rfp(grid,isGT){
  let s=0,r=0,f=0,p=0,land=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    if(isGT){const pr=grid[y][x];if(pr[0]>0.99||pr[5]>0.99)continue;land++;s+=(pr[1]||0)+(pr[2]||0);r+=pr[3]||0;f+=pr[4]||0;p+=pr[0]||0;}
    else{const t=grid[y][x];if(t===10||t===5)continue;land++;const c=t2c(t);if(c===1||c===2)s++;else if(c===3)r++;else if(c===4)f++;else if(c===0)p++;}}
  if(!land)return{sett:0,ruin:0,forest:0,plains:0};
  return{sett:s/land,ruin:r/land,forest:f/land,plains:p/land};
}
function rdist(a,b){return Math.sqrt((a.sett-b.sett)**2+(a.ruin-b.ruin)**2+(a.forest-b.forest)**2+(a.plains-b.plains)**2);}

// Preload ALL data
console.log('Loading data...');
const allInits={},allGT={},allReplays={},fps={};
const roundNums=[];
for(let r=1;r<=20;r++){
  const rn=`R${r}`;
  const ip=path.join(DD,`inits_${rn}.json`),gp=path.join(DD,`gt_${rn}.json`);
  if(!fs.existsSync(ip)||!fs.existsSync(gp))continue;
  allInits[rn]=JSON.parse(fs.readFileSync(ip));
  allGT[rn]=JSON.parse(fs.readFileSync(gp));
  const rp=path.join(DD,`replays_${rn}.json`);
  allReplays[rn]=fs.existsSync(rp)?JSON.parse(fs.readFileSync(rp)):[];
  // Fingerprint from GT
  const seedFPs=allGT[rn].map(g=>rfp(g,true));
  fps[rn]={sett:seedFPs.reduce((a,f)=>a+f.sett,0)/SEEDS,ruin:seedFPs.reduce((a,f)=>a+f.ruin,0)/SEEDS,
    forest:seedFPs.reduce((a,f)=>a+f.forest,0)/SEEDS,plains:seedFPs.reduce((a,f)=>a+f.plains,0)/SEEDS};
  roundNums.push(rn);
}
console.log(`Loaded ${roundNums.length} rounds: ${roundNums.join(', ')}`);
for(const rn of roundNums)console.log(`  ${rn}: sett=${fps[rn].sett.toFixed(3)} ruin=${fps[rn].ruin.toFixed(3)} forest=${fps[rn].forest.toFixed(3)} plains=${fps[rn].plains.toFixed(3)} replays=${allReplays[rn].length}`);

function buildModel(exclude, weights){
  const model={};
  for(const level of['d0','d1','d2','d3','d4']){
    for(const rn of roundNums){
      if(rn===exclude)continue;
      const rw=(weights&&weights[rn]!==undefined)?weights[rn]:1.0;
      if(rw<0.001)continue;
      for(let si=0;si<SEEDS;si++){
        if(!allInits[rn][si]||!allGT[rn][si])continue;
        const g=Array.isArray(allInits[rn][si].grid)?allInits[rn][si].grid:allInits[rn][si];
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
          const keys=fk(g,y,x);if(!keys)continue;const k=keys[level];
          if(!model[k])model[k]={n:0,counts:new Array(C).fill(0)};
          for(let c=0;c<C;c++)model[k].counts[c]+=allGT[rn][si][y][x][c]*20*rw;
          model[k].n+=20*rw;}}
      // Replays
      for(const rep of allReplays[rn]){
        const g=allInits[rn][rep.si]?(Array.isArray(allInits[rn][rep.si].grid)?allInits[rn][rep.si].grid:allInits[rn][rep.si]):null;
        if(!g)continue;
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
          const keys=fk(g,y,x);if(!keys)continue;const k=keys[level];
          if(!model[k])model[k]={n:0,counts:new Array(C).fill(0)};
          model[k].n+=rw;model[k].counts[t2c(rep.finalGrid[y][x])]+=rw;}}
    }
  }
  for(const k of Object.keys(model)){
    const tot=model[k].counts.reduce((a,b)=>a+b,0)+C*0.05;
    model[k].a=model[k].counts.map(v=>(v+0.05)/tot);}
  return model;
}

function predict(model,grid,TEMP=0.9){
  const pred=[];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const t=grid[y][x];
    if(t===10){pred[y][x]=[1,0,0,0,0,0];continue;}
    if(t===5){pred[y][x]=[0,0,0,0,0,1];continue;}
    const keys=fk(grid,y,x);
    if(!keys){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
    const lvls=['d0','d1','d2','d3','d4'],ws=[1,0.3,0.15,0.08,0.02];
    const p=[0,0,0,0,0,0];let wS=0;
    for(let li=0;li<lvls.length;li++){const d=model[keys[lvls[li]]];
      if(d&&d.n>=1){const w=ws[li]*Math.pow(d.n,0.5);for(let c=0;c<C;c++)p[c]+=w*d.a[c];wS+=w;}}
    if(wS===0){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
    let s=0;for(let c=0;c<C;c++){p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/TEMP);s+=p[c];}
    let s2=0;for(let c=0;c<C;c++){p[c]=Math.max(p[c]/s,0.0001);s2+=p[c];}
    for(let c=0;c<C;c++)p[c]/=s2;
    pred[y][x]=p;
  }}return pred;
}

function scoreFn(pred,gt){
  let wklN=0,wklD=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const gc=gt[y][x];let ent=0;
    for(let c=0;c<C;c++){if(gc[c]>0.001)ent-=gc[c]*Math.log(gc[c]);}
    if(ent<0.01)continue;
    let kl=0;for(let c=0;c<C;c++){if(gc[c]>0.001)kl+=gc[c]*Math.log(gc[c]/Math.max(pred[y][x][c],1e-10));}
    wklN+=ent*kl;wklD+=ent;}
  const wkl=wklD>0?wklN/wklD:0;
  return Math.max(0,Math.min(100,100*Math.exp(-3*wkl)));
}

// Test equal weights first (baseline)
console.log('\n=== Equal weights (baseline) ===');
const eqScores=[];
for(const testRn of roundNums){
  const seeds=[];
  const model=buildModel(testRn,null);
  for(let si=0;si<SEEDS;si++){
    if(!allInits[testRn][si]||!allGT[testRn][si])continue;
    const g=Array.isArray(allInits[testRn][si].grid)?allInits[testRn][si].grid:allInits[testRn][si];
    seeds.push(scoreFn(predict(model,g),allGT[testRn][si]));
  }
  const avg=seeds.reduce((a,b)=>a+b,0)/seeds.length;
  eqScores.push({rn:testRn,avg});
}
const eqAvg=eqScores.reduce((a,s)=>a+s.avg,0)/eqScores.length;
console.log(`Equal: avg=${eqAvg.toFixed(1)} | ${eqScores.map(s=>s.rn+'='+s.avg.toFixed(0)).join(' ')}`);

// Test regime-weighted
for(const sigma of [0.03, 0.05, 0.08, 0.12, 0.2]){
  const allS=[];
  for(const testRn of roundNums){
    const otherFPs={};for(const[k,v]of Object.entries(fps)){if(k!==testRn)otherFPs[k]=v;}
    const weights={};
    for(const[k,fp]of Object.entries(otherFPs)){const d=rdist(fps[testRn],fp);weights[k]=Math.exp(-d*d/(2*sigma*sigma));}
    const model=buildModel(testRn,weights);
    const seeds=[];
    for(let si=0;si<SEEDS;si++){
      if(!allInits[testRn][si]||!allGT[testRn][si])continue;
      const g=Array.isArray(allInits[testRn][si].grid)?allInits[testRn][si].grid:allInits[testRn][si];
      seeds.push(scoreFn(predict(model,g),allGT[testRn][si]));
    }
    allS.push({rn:testRn,avg:seeds.reduce((a,b)=>a+b,0)/seeds.length});
  }
  const avg=allS.reduce((a,s)=>a+s.avg,0)/allS.length;
  console.log(`sigma=${sigma}: avg=${avg.toFixed(1)} | ${allS.map(s=>s.rn+'='+s.avg.toFixed(0)).join(' ')}`);
}
