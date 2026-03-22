
const { H, W, selectClosestRounds } = require('./shared');
function mergeBkts(p,r){const m={};for(const n of r){const b=p[String(n)];if(!b)continue;for(const[k,v]of Object.entries(b)){if(!m[k])m[k]={count:0,sum:[0,0,0,0,0,0]};m[k].count+=v.count;for(let c=0;c<6;c++)m[k].sum[c]+=v.sum[c];}}const o={};for(const[k,v]of Object.entries(m))o[k]=v.sum.map(s=>s/v.count);return o;}
function predict(grid,setts,prb,gr,tr,cfg){
  const K=cfg.K||3,floor=cfg.FLOOR||0.0001;
  const tg=gr[String(tr)]||0.15;
  const cand={...gr};delete cand[String(tr)];
  const cr=selectClosestRounds(cand,tg,K);
  const ar=Object.keys(prb).map(Number).filter(n=>n!==tr);
  const fm=mergeBkts(prb,cr), am=mergeBkts(prb,ar);
  const sp=new Set(),sl=[];
  for(const s of setts){sp.add(s.y*W+s.x);sl.push(s);}
  const pred=[];
  for(let y=0;y<H;y++){const row=[];for(let x=0;x<W;x++){
    const v=grid[y][x];
    let std,rich;
    if(v===10){std='O';rich='O';}
    else if(v===5){std='M';rich='M';}
    else{
      const t=v===4?'F':(v===1||v===2)?'S':'P';
      let nS3=0,nS6=0;
      for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){
        if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;
        if(ny>=0&&ny<H&&nx>=0&&nx<W&&sp.has(ny*W+nx)){
          if(Math.abs(dy)<=3&&Math.abs(dx)<=3)nS3++;
          nS6++;
        }
      }
      let coast=false;
      for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){
        const ny=y+dy,nx=x+dx;
        if(ny>=0&&ny<H&&nx>=0&&nx<W&&grid[ny][nx]===10)coast=true;
      }
      std=t+(nS3===0?'0':nS3<=2?'1':nS3<=5?'2':'3')+(coast?'c':'');
      rich=std;
      if(nS3===0&&t!=='S'){
        const b=nS6===0?'z':nS6<=2?'y':nS6<=5?'x':'w';
        rich=std+b;
      }else if(nS3>=1&&nS3<=2&&t!=='S'){
        let md=999;
        for(const s of sl){const d=Math.max(Math.abs(s.y-y),Math.abs(s.x-x));if(d<=3)md=Math.min(md,d);}
        rich=std+(md<=1?'a':md<=2?'b':'');
      }
    }
    let rp=(rich!==std)?(fm[rich]||am[rich]):null;
    let sp2=fm[std]||am[std];
    if(!sp2){const fb=std.slice(0,-1);sp2=fm[fb]||am[fb];}
    if(!sp2)sp2=[1/6,1/6,1/6,1/6,1/6,1/6];
    const d=rp?rp:sp2;
    const fl=d.map(v=>Math.max(v,floor));
    const s=fl.reduce((a,b)=>a+b,0);
    row.push(fl.map(v=>v/s));
  }pred.push(row);}return pred;
}
module.exports={predict};