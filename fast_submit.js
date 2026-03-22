const https=require('https'),fs=require('fs'),path=require('path');
const H=40,W=40,SEEDS=5,C=6,DD=path.join(__dirname,'data');
const TOKEN=process.argv[2]||'';
const BASE='https://api.ainm.no/astar-island';
function api(m,p,b){return new Promise((res,rej)=>{const u=new URL(BASE+p);const pl=b?JSON.stringify(b):null;
  const o={hostname:u.hostname,path:u.pathname+u.search,method:m,headers:{'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'}};
  if(pl)o.headers['Content-Length']=Buffer.byteLength(pl);
  const r=https.request(o,re=>{let d='';re.on('data',c=>d+=c);re.on('end',()=>{try{res({ok:re.statusCode<300,status:re.statusCode,data:JSON.parse(d)});}catch{res({ok:false,status:re.statusCode,data:d});}});});
  r.on('error',rej);if(pl)r.write(pl);r.end();});}
const GET=p=>api('GET',p),POST=(p,b)=>api('POST',p,b);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

function t2c(t){return(t===10||t===11||t===0)?0:(t>=1&&t<=5)?t:0;}
function cf(g,y,x){const t=g[y][x];if(t===10||t===5)return null;
  let nS=0,co=0,fN=0,sR2=0;
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return{d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`,d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,d2:`D2_${t}_${sa>0?1:0}_${co}`,d3:`D3_${t}_${co}`,d4:`D4_${t}`};}
function bGT(gts,inits,rounds,level,alpha){const m={};for(const rn of rounds){if(!gts[rn]||!inits[rn])continue;for(let si=0;si<SEEDS;si++){if(!inits[rn][si]||!gts[rn][si])continue;for(let y=0;y<H;y++)for(let x=0;x<W;x++){const keys=cf(inits[rn][si],y,x);if(!keys)continue;const k=keys[level];if(!m[k])m[k]={n:0,counts:new Float64Array(C)};const p=gts[rn][si][y][x];for(let c=0;c<C;c++)m[k].counts[c]+=p[c];m[k].n++;}}}for(const k of Object.keys(m)){const tot=Array.from(m[k].counts).reduce((a,b)=>a+b,0)+C*alpha;m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/tot);}return m;}
function bRep(reps,inits,rounds,level,alpha){const m={};for(const rn of rounds){if(!reps[rn]||!inits[rn])continue;for(const rep of reps[rn]){const g=inits[rn][rep.si];if(!g)continue;for(let y=0;y<H;y++)for(let x=0;x<W;x++){const keys=cf(g,y,x);if(!keys)continue;const k=keys[level];const fc=t2c(rep.finalGrid[y][x]);if(!m[k])m[k]={n:0,counts:new Float64Array(C)};m[k].n++;m[k].counts[fc]++;}}}for(const k of Object.keys(m)){const tot=m[k].n+C*alpha;m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/tot);}return m;}
function merge(gtM,repM,gtW,alpha){const m={};const all=new Set([...Object.keys(gtM),...Object.keys(repM)]);for(const k of all){const gm=gtM[k],rm=repM[k];if(gm&&rm){const c=new Float64Array(C);for(let i=0;i<C;i++)c[i]=rm.counts[i]+gm.counts[i]*gtW;const t=Array.from(c).reduce((a,b)=>a+b,0)+C*alpha;m[k]={n:rm.n+gm.n*gtW,counts:c,a:Array.from(c).map(v=>(v+alpha)/t)};}else if(gm){const c=new Float64Array(C);for(let i=0;i<C;i++)c[i]=gm.counts[i]*gtW;const t=Array.from(c).reduce((a,b)=>a+b,0)+C*alpha;m[k]={n:gm.n*gtW,counts:c,a:Array.from(c).map(v=>(v+alpha)/t)};}else{m[k]={n:rm.n,counts:rm.counts,a:rm.a.slice()};}}return m;}
function predict(grid,model,temp){
  const fl=0.00005,pred=[];
  const hierW=[1.0,0.3,0.15,0.08,0.02];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const t=grid[y][x];if(t===10){pred[y][x]=[1,0,0,0,0,0];continue;}if(t===5){pred[y][x]=[0,0,0,0,0,1];continue;}
    const keys=cf(grid,y,x);if(!keys){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
    const levels=['d0','d1','d2','d3','d4'];const p=[0,0,0,0,0,0];let wS=0;
    for(let li=0;li<levels.length;li++){const d=model[keys[levels[li]]];if(d&&d.n>=1){const w=hierW[li]*Math.pow(d.n,0.5);for(let c=0;c<C;c++)p[c]+=w*d.a[c];wS+=w;}}
    if(wS===0){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
    let s=0;for(let c=0;c<C;c++){p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/temp);if(p[c]<fl)p[c]=fl;s+=p[c];}
    for(let c=0;c<C;c++)p[c]/=s;pred[y][x]=p;
  }}return pred;
}
function score(pred,gt){let tE=0,tWK=0;for(let y=0;y<H;y++)for(let x=0;x<W;x++){const p=gt[y][x],q=pred[y][x];let e=0;for(let c=0;c<C;c++)if(p[c]>0.001)e-=p[c]*Math.log(p[c]);if(e<0.01)continue;let kl=0;for(let c=0;c<C;c++)if(p[c]>0.001)kl+=p[c]*Math.log(p[c]/Math.max(q[c],1e-10));tE+=e;tWK+=e*kl;}if(tE===0)return 100;return Math.max(0,Math.min(100,100*Math.exp(-3*tWK/tE)));}

async function main(){
  console.log('=== FAST LOO + SUBMIT ===');
  const I={},G={},R={};const TR=[];
  for(let r=1;r<=6;r++){if(r===3)continue;const rn=`R${r}`;
    if(fs.existsSync(path.join(DD,`inits_${rn}.json`)))I[rn]=JSON.parse(fs.readFileSync(path.join(DD,`inits_${rn}.json`)));
    if(fs.existsSync(path.join(DD,`gt_${rn}.json`)))G[rn]=JSON.parse(fs.readFileSync(path.join(DD,`gt_${rn}.json`)));
    if(fs.existsSync(path.join(DD,`replays_${rn}.json`)))R[rn]=JSON.parse(fs.readFileSync(path.join(DD,`replays_${rn}.json`)));
    if(I[rn]&&G[rn])TR.push(rn);
  }
  console.log('Rounds:',TR.join(', '),'Replays:',TR.filter(r=>R[r]).map(r=>`${r}=${R[r].length}`).join(', '));

  // Full LOO with gtW=10, temp=1.05
  console.log('\n=== Full LOO with gtW=10, temp=1.05 ===');
  const gtWs=[3,5,10,20];
  const temps=[1.0,1.05,1.1];
  
  let bestAvg=0,bestGtW=3,bestTemp=1.05;
  for(const gtW of gtWs){
    for(const temp of temps){
      const allScores=[];
      for(const testR of TR){
        const others=TR.filter(r=>r!==testR);
        const model={};
        for(const level of['d0','d1','d2','d3','d4']){
          const gtM=bGT(G,I,others,level,0.05);
          const repRounds=others.filter(r=>R[r]);
          const repM=repRounds.length>0?bRep(R,I,repRounds,level,0.05):{};
          const merged=merge(gtM,repM,gtW,0.05);
          for(const[k,v]of Object.entries(merged))if(!model[k])model[k]=v;
        }
        for(let si=0;si<SEEDS;si++){
          if(!I[testR][si]||!G[testR][si])continue;
          allScores.push(score(predict(I[testR][si],model,temp),G[testR][si]));
        }
      }
      const avg=allScores.reduce((a,b)=>a+b,0)/allScores.length;
      const perRound={};
      let idx=0;
      for(const testR of TR){for(let si=0;si<SEEDS;si++){if(!I[testR][si]||!G[testR][si])continue;if(!perRound[testR])perRound[testR]=[];perRound[testR].push(allScores[idx++]);}}
      const roundAvgs=Object.entries(perRound).map(([r,s])=>`${r}=${(s.reduce((a,b)=>a+b,0)/s.length).toFixed(1)}`).join(' ');
      console.log(`  gtW=${gtW} temp=${temp}: avg=${avg.toFixed(3)} [${roundAvgs}]`);
      if(avg>bestAvg){bestAvg=avg;bestGtW=gtW;bestTemp=temp;}
    }
  }
  console.log(`\nBest: gtW=${bestGtW} temp=${bestTemp} → ${bestAvg.toFixed(3)}`);

  // Build FINAL model with ALL rounds and best config
  console.log('\nBuilding final model with ALL rounds...');
  const finalModel={};
  for(const level of['d0','d1','d2','d3','d4']){
    const gtM=bGT(G,I,TR,level,0.05);
    const repRounds=TR.filter(r=>R[r]);
    const repM=repRounds.length>0?bRep(R,I,repRounds,level,0.05):{};
    const merged=merge(gtM,repM,bestGtW,0.05);
    for(const[k,v]of Object.entries(merged))if(!finalModel[k])finalModel[k]=v;
  }
  console.log(`Final model: ${Object.keys(finalModel).length} keys`);

  // Load R7 inits and submit seed 4
  const R7_ID='36e581f1-73f8-453f-ab98-cbe3052b701b';
  const{data:r7D}=await GET('/rounds/'+R7_ID);
  const r7I=r7D.initial_states.map(is=>is.grid);

  console.log('\n=== Submitting R7 seed 4 ===');
  const p4=predict(r7I[4],finalModel,bestTemp);
  let valid=true;
  for(let y=0;y<H&&valid;y++)for(let x=0;x<W&&valid;x++){const s=p4[y][x].reduce((a,b)=>a+b,0);if(Math.abs(s-1)>0.02)valid=false;}
  if(valid){
    const res=await POST('/submit',{round_id:R7_ID,seed_index:4,prediction:p4});
    console.log(`Seed 4: ${res.ok?'✅':'❌'} ${JSON.stringify(res.data).slice(0,80)}`);
  }else{console.log('VALIDATION FAILED');}
  
  console.log(`\nExpected R7 seed 4 score: ~${bestAvg.toFixed(1)}`);
  console.log(`R7 weight: 1.4071`);
  console.log(`ws from this seed alone would be: ${(bestAvg*1.4071).toFixed(1)}`);
  console.log('\nDONE. Seeds 0-3 kept with viewport-enhanced predictions.');
}
main().catch(e=>console.error('Error:',e.message,e.stack));
