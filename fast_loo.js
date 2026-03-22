const fs=require('fs'),path=require('path');
const H=40,W=40,SEEDS=5,C=6;
const DD=path.join(__dirname,'data');
function t2c(t){return(t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0;}
function cf(g,y,x){
  const t=g[y][x];if(t===10||t===5)return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;
  }
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;
  }
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return {d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`,d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,d2:`D2_${t}_${sa>0?1:0}_${co}`,d3:`D3_${t}_${co}`,d4:`D4_${t}`};
}
function bGT(gts,inits,rounds,level,alpha){
  const m={};for(const rn of rounds){if(!gts[rn]||!inits[rn])continue;
    for(let si=0;si<SEEDS;si++){if(!inits[rn][si]||!gts[rn][si])continue;
      for(let y=0;y<H;y++)for(let x=0;x<W;x++){
        const keys=cf(inits[rn][si],y,x);if(!keys)continue;const k=keys[level];
        if(!m[k])m[k]={n:0,counts:new Float64Array(C)};
        const p=gts[rn][si][y][x];for(let c=0;c<C;c++)m[k].counts[c]+=p[c];m[k].n++;
  }}}
  for(const k of Object.keys(m)){const tot=Array.from(m[k].counts).reduce((a,b)=>a+b,0)+C*alpha;
    m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/tot);}
  return m;
}
function bRep(reps,inits,rounds,level,alpha){
  const m={};for(const rn of rounds){if(!reps[rn]||!inits[rn])continue;
    for(const rep of reps[rn]){const g=inits[rn][rep.si];if(!g)continue;
      for(let y=0;y<H;y++)for(let x=0;x<W;x++){
        const keys=cf(g,y,x);if(!keys)continue;const k=keys[level];
        const fc=t2c(rep.finalGrid[y][x]);
        if(!m[k])m[k]={n:0,counts:new Float64Array(C)};m[k].n++;m[k].counts[fc]++;
  }}}
  for(const k of Object.keys(m)){const tot=m[k].n+C*alpha;
    m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/tot);}
  return m;
}
function merge(gtM,repM,gtW,alpha){
  const m={};const all=new Set([...Object.keys(gtM),...Object.keys(repM)]);
  for(const k of all){const gm=gtM[k],rm=repM[k];
    if(gm&&rm){const c=new Float64Array(C);for(let i=0;i<C;i++)c[i]=rm.counts[i]+gm.counts[i]*gtW;
      const t=Array.from(c).reduce((a,b)=>a+b,0)+C*alpha;m[k]={n:rm.n+gm.n*gtW,counts:c,a:Array.from(c).map(v=>(v+alpha)/t)};}
    else if(gm){const c=new Float64Array(C);for(let i=0;i<C;i++)c[i]=gm.counts[i]*gtW;
      const t=Array.from(c).reduce((a,b)=>a+b,0)+C*alpha;m[k]={n:gm.n*gtW,counts:c,a:Array.from(c).map(v=>(v+alpha)/t)};}
    else{m[k]={n:rm.n,counts:rm.counts,a:rm.a.slice()};}
  }
  return m;
}
// Predict with configurable hierarchy weights and temperature
function predict(grid,model,hierW,temp,fl){
  fl=fl||0.00005;const pred=[];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const t=grid[y][x];
    if(t===10){pred[y][x]=[1,0,0,0,0,0];continue;}
    if(t===5){pred[y][x]=[0,0,0,0,0,1];continue;}
    const keys=cf(grid,y,x);
    if(!keys){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
    const levels=['d0','d1','d2','d3','d4'];
    const p=[0,0,0,0,0,0];let wS=0;
    for(let li=0;li<levels.length;li++){
      const d=model[keys[levels[li]]];
      if(d&&d.n>=1){const w=hierW[li]*Math.pow(d.n,0.5);
        for(let c=0;c<C;c++)p[c]+=w*d.a[c];wS+=w;}
    }
    if(wS===0){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
    let s=0;for(let c=0;c<C;c++){p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/temp);if(p[c]<fl)p[c]=fl;s+=p[c];}
    for(let c=0;c<C;c++)p[c]/=s;pred[y][x]=p;
  }}
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
const I={},G={},R={};const TR=[];
for(let r=1;r<=6;r++){if(r===3)continue;const rn=`R${r}`;
  const iF=path.join(DD,`inits_${rn}.json`),gF=path.join(DD,`gt_${rn}.json`),rF=path.join(DD,`replays_${rn}.json`);
  if(fs.existsSync(iF))I[rn]=JSON.parse(fs.readFileSync(iF));
  if(fs.existsSync(gF))G[rn]=JSON.parse(fs.readFileSync(gF));
  if(fs.existsSync(rF))R[rn]=JSON.parse(fs.readFileSync(rF));
  if(I[rn]&&G[rn])TR.push(rn);
}
console.log('Rounds:',TR.join(', '));
console.log('Replays:',TR.filter(r=>R[r]).map(r=>`${r}=${R[r].length}`).join(', '));

// Pre-build LOO models (5 folds)
console.log('\nPre-building LOO models...');
const looModels={};
for(const testR of TR){
  const others=TR.filter(r=>r!==testR);
  const model={};
  for(const level of['d0','d1','d2','d3','d4']){
    const gtM=bGT(G,I,others,level,0.05);
    const repRounds=others.filter(r=>R[r]);
    const repM=repRounds.length>0?bRep(R,I,repRounds,level,0.05):{};
    const merged=merge(gtM,repM,3,0.05);
    for(const[k,v]of Object.entries(merged))if(!model[k])model[k]=v;
  }
  looModels[testR]=model;
}
console.log('Done building', Object.keys(looModels).length, 'LOO models');

// Test hierarchy configs × temperatures
console.log('\n=== Testing Hierarchy + Temperature ===');
const hierConfigs=[
  {name:'orig',    w:[1.0,0.3,0.15,0.08,0.02]},
  {name:'D1-heavy',w:[0.3,1.0,0.15,0.08,0.02]},
  {name:'D1-only', w:[0.0,1.0,0.0,0.0,0.0]},
  {name:'D0+D1',   w:[1.0,1.0,0.15,0.08,0.02]},
  {name:'flat',    w:[1.0,1.0,1.0,1.0,1.0]},
  {name:'steep',   w:[1.0,0.1,0.01,0.001,0.0001]},
];
const temps=[0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.2,1.3,1.5,2.0];

let globalBest=0,globalBestCfg=null;
const results=[];

for(const hc of hierConfigs){
  for(const temp of temps){
    const scores=[];
    for(const testR of TR){
      for(let si=0;si<SEEDS;si++){
        if(!I[testR][si]||!G[testR][si])continue;
        scores.push(score(predict(I[testR][si],looModels[testR],hc.w,temp),G[testR][si]));
      }
    }
    const avg=scores.reduce((a,b)=>a+b,0)/scores.length;
    const min=Math.min(...scores);
    results.push({name:hc.name,temp,avg,min});
    if(avg>globalBest){globalBest=avg;globalBestCfg={hier:hc,temp};}
  }
}

// Sort by avg score and print top 20
results.sort((a,b)=>b.avg-a.avg);
console.log('\nTop 20 configs:');
for(let i=0;i<Math.min(20,results.length);i++){
  const r=results[i];
  console.log(`  ${r.name.padEnd(10)} temp=${r.temp.toFixed(2)} avg=${r.avg.toFixed(3)} min=${r.min.toFixed(1)}`);
}

console.log(`\n🏆 BEST: ${globalBestCfg.hier.name} temp=${globalBestCfg.temp} → ${globalBest.toFixed(3)}`);
console.log(`   Expected R7 ws: ${(globalBest * 1.4071).toFixed(1)} (cross-round only)`);

// Also test different GT weights
console.log('\n=== Testing GT weights ===');
for(const gtW of [1,2,3,5,10]){
  const model={};
  for(const level of['d0','d1','d2','d3','d4']){
    const gtM=bGT(G,I,TR.filter(r=>r!==TR[0]),level,0.05);
    const repRounds=TR.filter(r=>r!==TR[0]&&R[r]);
    const repM=repRounds.length>0?bRep(R,I,repRounds,level,0.05):{};
    const merged=merge(gtM,repM,gtW,0.05);
    for(const[k,v]of Object.entries(merged))if(!model[k])model[k]=v;
  }
  const s=score(predict(I[TR[0]][0],model,globalBestCfg.hier.w,globalBestCfg.temp),G[TR[0]][0]);
  console.log(`  gtW=${gtW}: ${s.toFixed(3)} (first fold only)`);
}

console.log('\nBest config to use for R7 submission:');
console.log(`  Hierarchy: ${globalBestCfg.hier.name} ${JSON.stringify(globalBestCfg.hier.w)}`);
console.log(`  Temperature: ${globalBestCfg.temp}`);
console.log(`  LOO: ${globalBest.toFixed(3)}`);
