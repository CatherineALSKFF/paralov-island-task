// R2 Monte Carlo Solver — Paste in browser console on app.ainm.no
// 1. Collects R1 replays to learn simulator rules
// 2. Runs MC simulation on all 5 R2 seeds
// 3. Submits predictions (resubmit overwrites previous)
(async function() {
  var B='https://api.ainm.no/astar-island';
  var R1='71451d74-be9f-471f-aacd-a41f3b68a9cd';
  var R2='76909e29-f664-4b2f-b16b-61b7507277e9';
  var H=40,W=40,SEEDS=5;
  var NREP=80, NSIM=500; // replays to collect, MC sims per seed
  function t2c(t){return(t===10||t===11||t===0)?0:((t>=1&&t<=5)?t:0);}
  function log(m){console.log('[MC] '+m); document.title=m;}

  // ═══ PHASE 1: Collect R1 replays ═══
  log('Collecting '+NREP+' R1 replays...');
  var reps=[];
  for(var i=0;i<NREP;i++){
    try{
      var r=await fetch(B+'/replay',{method:'POST',credentials:'include',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({round_id:R1,seed_index:i%5})});
      var d=await r.json();
      if(d&&d.frames&&d.frames.length>=51) reps.push(d);
    }catch(e){}
    if(i%2===1) await new Promise(function(r){setTimeout(r,80);});
    if(i%20===19) log('Replays: '+reps.length+'/'+(i+1));
  }
  log('Got '+reps.length+' replays');

  // ═══ PHASE 2: Learn per-step transitions ═══
  log('Learning transitions...');
  var TT={};
  for(var ri=0;ri<reps.length;ri++){
    var frames=reps[ri].frames;
    for(var f=0;f<50;f++){
      var curr=frames[f].grid, next=frames[f+1].grid;
      for(var y=0;y<H;y++) for(var x=0;x<W;x++){
        var s=curr[y][x]; if(s===10||s===5) continue;
        var nO=0,nS=0,nF=0;
        for(var dy=-1;dy<=1;dy++) for(var dx=-1;dx<=1;dx++){
          if(!dy&&!dx) continue;var ny=y+dy,nx=x+dx;
          if(ny<0||ny>=H||nx<0||nx>=W){nO++;continue;}
          var t=curr[ny][nx];if(t===10)nO++;else if(t===1||t===2)nS++;else if(t===4)nF++;
        }
        var ph=Math.min(4,Math.floor(f/10));
        var dk=ph+'_'+s+'_'+nS+'_'+nO+'_'+nF;
        if(!TT[dk])TT[dk]={};TT[dk][next[y][x]]=(TT[dk][next[y][x]]||0)+1;
        var ck='C_'+ph+'_'+s+'_'+(nS>0?1:0)+'_'+(nO>0?1:0);
        if(!TT[ck])TT[ck]={};TT[ck][next[y][x]]=(TT[ck][next[y][x]]||0)+1;
      }
    }
  }
  var TP={};
  for(var k in TT){var tot=0;for(var s in TT[k])tot+=TT[k][s];
    TP[k]={_n:tot};for(var s in TT[k])TP[k][parseInt(s)]=TT[k][s]/tot;}
  log(Object.keys(TP).length+' transition keys');

  // ═══ PHASE 3: Also load R1 GT for blending ═══
  log('Loading R1 GT transitions...');
  var trans={},tCnt={};
  var r1det=await(await fetch(B+'/rounds/'+R1,{credentials:'include'})).json();
  for(var gs=0;gs<5;gs++){
    var an=await(await fetch(B+'/analysis/'+R1+'/'+gs,{credentials:'include'})).json();
    for(var y=0;y<H;y++) for(var x=0;x<W;x++){
      var key=r1det.initial_states[gs].grid[y][x];
      if(!trans[key]){trans[key]=[0,0,0,0,0,0];tCnt[key]=0;}
      for(var c=0;c<6;c++) trans[key][c]+=an.ground_truth[y][x][c];
      tCnt[key]++;
    }
  }
  for(var key in trans){var n=tCnt[key];for(var c=0;c<6;c++) trans[key][c]/=n;}

  // Also build 3x3 neighborhood index from R1 GT
  var nbrIdx={};
  function nbrHash(grid,y,x){
    var h=''+grid[y][x];
    var dirs=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
    for(var d=0;d<8;d++){var ny=y+dirs[d][0],nx=x+dirs[d][1];
      h+=','+(ny>=0&&ny<H&&nx>=0&&nx<W?grid[ny][nx]:10);}
    return h;
  }
  for(var gs=0;gs<5;gs++){
    var an=await(await fetch(B+'/analysis/'+R1+'/'+gs,{credentials:'include'})).json();
    var grid=r1det.initial_states[gs].grid;
    for(var y=0;y<H;y++) for(var x=0;x<W;x++){
      var h=nbrHash(grid,y,x);
      if(!nbrIdx[h]) nbrIdx[h]={sum:[0,0,0,0,0,0],n:0};
      for(var c=0;c<6;c++) nbrIdx[h].sum[c]+=an.ground_truth[y][x][c];
      nbrIdx[h].n++;
    }
  }
  log('R1 GT loaded: '+Object.keys(nbrIdx).length+' neighborhood patterns');

  // ═══ PHASE 4: MC simulate + blend + submit each R2 seed ═══
  var r2det=await(await fetch(B+'/rounds/'+R2,{credentials:'include'})).json();

  function simOne(ig){
    var g=[];for(var y=0;y<H;y++){g[y]=[];for(var x=0;x<W;x++) g[y][x]=ig[y][x];}
    for(var st=0;st<50;st++){
      var ng=[];for(var y=0;y<H;y++){ng[y]=[];for(var x=0;x<W;x++) ng[y][x]=g[y][x];}
      var ph=Math.min(4,Math.floor(st/10));
      for(var y=0;y<H;y++) for(var x=0;x<W;x++){
        var s=g[y][x]; if(s===10||s===5) continue;
        var nO=0,nS=0,nF=0;
        for(var dy=-1;dy<=1;dy++) for(var dx=-1;dx<=1;dx++){
          if(!dy&&!dx) continue;var ny=y+dy,nx=x+dx;
          if(ny<0||ny>=H||nx<0||nx>=W){nO++;continue;}
          var t=g[ny][nx];if(t===10)nO++;else if(t===1||t===2)nS++;else if(t===4)nF++;
        }
        var dk=ph+'_'+s+'_'+nS+'_'+nO+'_'+nF;
        var pr=TP[dk]&&TP[dk]._n>=10?TP[dk]:null;
        if(!pr){var ck='C_'+ph+'_'+s+'_'+(nS>0?1:0)+'_'+(nO>0?1:0);pr=TP[ck];}
        if(!pr) continue;
        var r=Math.random(),cum=0;
        for(var ns in pr){if(ns==='_n')continue;cum+=pr[ns];if(r<cum){ng[y][x]=parseInt(ns);break;}}
      }
      g=ng;
    }
    return g;
  }

  for(var seed=0;seed<SEEDS;seed++){
    log('Simulating seed '+seed+' ('+NSIM+' MC runs)...');
    var counts=[];
    for(var y=0;y<H;y++){counts[y]=[];for(var x=0;x<W;x++) counts[y][x]=[0,0,0,0,0,0];}
    for(var sim=0;sim<NSIM;sim++){
      var final=simOne(r2det.initial_states[seed].grid);
      for(var y=0;y<H;y++) for(var x=0;x<W;x++) counts[y][x][t2c(final[y][x])]++;
      if(sim%100===99) log('Seed '+seed+': sim '+sim+'/'+NSIM);
    }

    // Build blended prediction: 50% MC + 50% R1 matching
    var pred=[];
    var grid=r2det.initial_states[seed].grid;
    for(var y=0;y<H;y++){pred[y]=[];for(var x=0;x<W;x++){
      // MC prediction
      var mc=[];for(var c=0;c<6;c++) mc[c]=counts[y][x][c]/NSIM;
      // R1 matching prediction
      var h=nbrHash(grid,y,x);
      var match=nbrIdx[h]&&nbrIdx[h].n>=2?nbrIdx[h]:null;
      var r1p;
      if(match){r1p=[];for(var c=0;c<6;c++) r1p[c]=match.sum[c]/match.n;}
      else{var key=grid[y][x];r1p=trans[key]?trans[key].slice():[0.167,0.167,0.167,0.167,0.167,0.167];}

      // Blend: weight MC more for dynamic cells, R1 more for static
      var mcEnt=0;for(var c=0;c<6;c++) if(mc[c]>0.001) mcEnt-=mc[c]*Math.log(mc[c]);
      var w_mc = mcEnt > 0.5 ? 0.6 : 0.3; // more MC weight for dynamic cells
      var w_r1 = 1 - w_mc;

      var p=[];var sum=0;
      for(var c=0;c<6;c++){
        p[c] = w_mc*mc[c] + w_r1*r1p[c] + 0.001; // tiny smoothing
        sum+=p[c];
      }
      var maxI=0,maxV=0;
      for(var c=0;c<6;c++){p[c]=parseFloat((p[c]/sum).toFixed(6));if(p[c]>maxV){maxV=p[c];maxI=c;}}
      sum=0;for(var c=0;c<6;c++) sum+=p[c];
      p[maxI]=parseFloat((p[maxI]+(1.0-sum)).toFixed(6));
      pred[y][x]=p;
    }}

    // Submit this seed
    log('Submitting seed '+seed+'...');
    var resp=await fetch(B+'/submit',{method:'POST',credentials:'include',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({round_id:R2,seed_index:seed,prediction:pred})});
    var res=await resp.json();
    log('Seed '+seed+': '+JSON.stringify(res));
    await new Promise(function(r){setTimeout(r,500);});
  }

  log('=== ALL 5 SEEDS SUBMITTED ===');
  var closes=new Date('2026-03-19T23:47:20.455071+00:00');
  var mins=Math.floor((closes-new Date())/60000);
  log(mins+' minutes remaining for further improvements');
  return 'Done! All 5 seeds submitted with MC+matching blend.';
})();
