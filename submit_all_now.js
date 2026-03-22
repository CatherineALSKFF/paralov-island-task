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
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(dy===0&&dx===0)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;const nt=g[ny][nx];
    if(nt===1||nt===2)nS++;if(nt===10)co=1;if(nt===4)fN++;}
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    if(Math.abs(dy)<=1&&Math.abs(dx)<=1)continue;const ny=y+dy,nx=x+dx;
    if(ny<0||ny>=H||nx<0||nx>=W)continue;if(g[ny][nx]===1||g[ny][nx]===2)sR2++;}
  const sa=Math.min(nS,5),sb2=sR2===0?0:sR2<=2?1:sR2<=4?2:3,fb=fN<=1?0:fN<=3?1:2;
  return{d0:`D0_${t}_${sa}_${co}_${sb2}_${fb}`,d1:`D1_${t}_${Math.min(sa,3)}_${co}_${sb2}`,
    d2:`D2_${t}_${sa>0?1:0}_${co}`,d3:`D3_${t}_${co}`,d4:`D4_${t}`};}
function bGT(gts,inits,rounds,level,alpha){const m={};for(const rn of rounds){if(!gts[rn]||!inits[rn])continue;for(let si=0;si<SEEDS;si++){if(!inits[rn][si]||!gts[rn][si])continue;for(let y=0;y<H;y++)for(let x=0;x<W;x++){const keys=cf(inits[rn][si],y,x);if(!keys)continue;const k=keys[level];if(!m[k])m[k]={n:0,counts:new Float64Array(C)};const p=gts[rn][si][y][x];for(let c=0;c<C;c++)m[k].counts[c]+=p[c];m[k].n++;}}}for(const k of Object.keys(m)){const tot=Array.from(m[k].counts).reduce((a,b)=>a+b,0)+C*alpha;m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/tot);}return m;}
function bRep(reps,inits,rounds,level,alpha){const m={};for(const rn of rounds){if(!reps[rn]||!inits[rn])continue;for(const rep of reps[rn]){const g=inits[rn][rep.si];if(!g)continue;for(let y=0;y<H;y++)for(let x=0;x<W;x++){const keys=cf(g,y,x);if(!keys)continue;const k=keys[level];const fc=t2c(rep.finalGrid[y][x]);if(!m[k])m[k]={n:0,counts:new Float64Array(C)};m[k].n++;m[k].counts[fc]++;}}}for(const k of Object.keys(m)){const tot=m[k].n+C*alpha;m[k].a=Array.from(m[k].counts).map(v=>(v+alpha)/tot);}return m;}

async function main(){
  console.log('=== SUBMITTING ALL 5 SEEDS ===');
  console.log('Time:', new Date().toISOString());
  const I={},G={},R={};const TR=[];
  for(let r=1;r<=6;r++){if(r===3)continue;const rn=`R${r}`;
    const iF=path.join(DD,`inits_${rn}.json`),gF=path.join(DD,`gt_${rn}.json`),rF=path.join(DD,`replays_${rn}.json`);
    if(fs.existsSync(iF))I[rn]=JSON.parse(fs.readFileSync(iF));
    if(fs.existsSync(gF))G[rn]=JSON.parse(fs.readFileSync(gF));
    if(fs.existsSync(rF))R[rn]=JSON.parse(fs.readFileSync(rF));
    if(I[rn]&&G[rn])TR.push(rn);
  }
  console.log('Training:',TR.join(', '));

  const gtW=20,temp=1.05,alpha=0.05;
  const model={};
  for(const level of['d0','d1','d2','d3','d4']){
    const gtM=bGT(G,I,TR,level,alpha);
    const repRounds=TR.filter(r=>R[r]);
    const repM=repRounds.length>0?bRep(R,I,repRounds,level,alpha):{};
    const allKeys=new Set([...Object.keys(gtM),...Object.keys(repM)]);
    for(const k of allKeys){
      const gm=gtM[k],rm=repM[k];
      if(gm&&rm){const c=new Float64Array(C);for(let i=0;i<C;i++)c[i]=rm.counts[i]+gm.counts[i]*gtW;
        const t=Array.from(c).reduce((a,b)=>a+b,0)+C*alpha;
        if(!model[k])model[k]={n:rm.n+gm.n*gtW,counts:c,a:Array.from(c).map(v=>(v+alpha)/t)};
      }else if(gm){const c=new Float64Array(C);for(let i=0;i<C;i++)c[i]=gm.counts[i]*gtW;
        const t=Array.from(c).reduce((a,b)=>a+b,0)+C*alpha;
        if(!model[k])model[k]={n:gm.n*gtW,counts:c,a:Array.from(c).map(v=>(v+alpha)/t)};
      }else{if(!model[k])model[k]={n:rm.n,counts:rm.counts,a:rm.a.slice()};}
    }
  }
  console.log('Model:',Object.keys(model).length,'keys, gtW='+gtW+' temp='+temp);

  const R7='36e581f1-73f8-453f-ab98-cbe3052b701b';
  const{data:rd}=await GET('/rounds/'+R7);
  const inits=rd.initial_states.map(is=>is.grid);

  for(let si=0;si<SEEDS;si++){
    const pred=[];
    for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
      const t=inits[si][y][x];
      if(t===10){pred[y][x]=[1,0,0,0,0,0];continue;}
      if(t===5){pred[y][x]=[0,0,0,0,0,1];continue;}
      const keys=cf(inits[si],y,x);
      if(!keys){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
      const levels=['d0','d1','d2','d3','d4'];
      const ws=[1.0,0.3,0.15,0.08,0.02];
      const p=[0,0,0,0,0,0];let wS=0;
      for(let li=0;li<levels.length;li++){
        const d=model[keys[levels[li]]];
        if(d&&d.n>=1){const w=ws[li]*Math.pow(d.n,0.5);for(let c=0;c<C;c++)p[c]+=w*d.a[c];wS+=w;}
      }
      if(wS===0){pred[y][x]=[1/6,1/6,1/6,1/6,1/6,1/6];continue;}
      let s=0;for(let c=0;c<C;c++){
        p[c]=Math.pow(Math.max(p[c]/wS,1e-10),1/temp);
        if(p[c]<0.00005)p[c]=0.00005;s+=p[c];}
      for(let c=0;c<C;c++)p[c]/=s;pred[y][x]=p;
    }}
    const res=await POST('/submit',{round_id:R7,seed_index:si,prediction:pred});
    console.log('Seed '+si+': '+(res.ok?'ACCEPTED':'FAILED')+' '+JSON.stringify(res.data).slice(0,80));
    await sleep(500);
  }
  console.log('\nALL 5 SEEDS SUBMITTED with optimized model');
  console.log('Expected score: ~80.6, ws: ~'+(80.6*1.4071).toFixed(0));
}
main().catch(e=>console.error('Error:',e.message,e.stack));
