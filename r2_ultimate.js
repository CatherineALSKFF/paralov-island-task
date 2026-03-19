// R2 ULTIMATE SOLVER — Paste in browser console on app.ainm.no
// Collects 100+ R1 replays, learns rules, MC simulates R2, blends with R1 GT, submits
// Progress shown in page title + console
(async function() {
  var B='https://api.ainm.no/astar-island';
  var R1='71451d74-be9f-471f-aacd-a41f3b68a9cd';
  var R2='76909e29-f664-4b2f-b16b-61b7507277e9';
  var H=40,W=40,SEEDS=5;
  function t2c(t){return(t===10||t===11||t===0)?0:((t>=1&&t<=5)?t:0);}
  function log(m){console.log('[R2U] '+m); document.title=m;}

  var closes=new Date('2026-03-19T23:47:20.455071+00:00');
  function minsLeft(){return Math.floor((closes-new Date())/60000);}
  log('Starting! '+minsLeft()+'min left');

  // ═══ PHASE 1: Collect R1 replays (aim for 100+) ═══
  log('Phase 1: Collecting R1 replays...');
  var reps=[];
  for(var i=0;i<120;i++){
    try{
      var r=await fetch(B+'/replay',{method:'POST',credentials:'include',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({round_id:R1,seed_index:i%5})});
      var d=await r.json();
      if(d&&d.frames&&d.frames.length>=51) reps.push(d);
    }catch(e){}
    if(i%3===2) await new Promise(function(r){setTimeout(r,80);});
    if(i%30===29) log('Replays: '+reps.length+'/'+(i+1)+' ('+minsLeft()+'min)');
  }
  log('Got '+reps.length+' replays. Building rules...');

  // ═══ PHASE 2: Learn detailed transition rules ═══
  var TT={};
  for(var ri=0;ri<reps.length;ri++){
    var frames=reps[ri].frames;
    for(var f=0;f<50;f++){
      var curr=frames[f].grid, next=frames[f+1].grid;
      for(var y=0;y<H;y++) for(var x=0;x<W;x++){
        var s=curr[y][x]; if(s===10||s===5) continue;
        var nO=0,nS=0,nF=0,nR=0;
        for(var dy=-1;dy<=1;dy++) for(var dx=-1;dx<=1;dx++){
          if(!dy&&!dx) continue;var ny=y+dy,nx=x+dx;
          if(ny<0||ny>=H||nx<0||nx>=W){nO++;continue;}
          var t=curr[ny][nx];
          if(t===10)nO++; else if(t===1||t===2)nS++; else if(t===4)nF++; else if(t===3)nR++;
        }
        var ph=Math.min(4,Math.floor(f/10));
        // Detailed key: phase + state + exact settlement count + ocean flag + forest flag
        var dk=ph+'_'+s+'_'+nS+'_'+(nO>0?1:0)+'_'+(nF>2?1:0);
        if(!TT[dk])TT[dk]={};TT[dk][next[y][x]]=(TT[dk][next[y][x]]||0)+1;
        // Coarse fallback
        var ck='C_'+ph+'_'+s+'_'+(nS>0?1:0)+'_'+(nO>0?1:0);
        if(!TT[ck])TT[ck]={};TT[ck][next[y][x]]=(TT[ck][next[y][x]]||0)+1;
        // Ultra-coarse
        var uk='U_'+ph+'_'+s;
        if(!TT[uk])TT[uk]={};TT[uk][next[y][x]]=(TT[uk][next[y][x]]||0)+1;
      }
    }
  }
  var TP={};
  for(var k in TT){var tot=0;for(var s in TT[k])tot+=TT[k][s];
    TP[k]={_n:tot};for(var s in TT[k])TP[k][parseInt(s)]=TT[k][s]/tot;}
  log(Object.keys(TP).length+' transition keys. Loading R1 GT...');

  // ═══ PHASE 3: Load R1 GT for blending ═══
  var trans={},tCnt={};
  var r1d=await(await fetch(B+'/rounds/'+R1,{credentials:'include'})).json();
  // Build 3x3 neighborhood index
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
    var grid=r1d.initial_states[gs].grid;
    for(var y=0;y<H;y++) for(var x=0;x<W;x++){
      // Terrain type average
      var key=grid[y][x];
      if(!trans[key]){trans[key]=[0,0,0,0,0,0];tCnt[key]=0;}
      for(var c=0;c<6;c++) trans[key][c]+=an.ground_truth[y][x][c];
      tCnt[key]++;
      // 3x3 neighborhood index
      var h=nbrHash(grid,y,x);
      if(!nbrIdx[h]) nbrIdx[h]={sum:[0,0,0,0,0,0],n:0};
      for(var c=0;c<6;c++) nbrIdx[h].sum[c]+=an.ground_truth[y][x][c];
      nbrIdx[h].n++;
    }
  }
  for(var key in trans){var n=tCnt[key];for(var c=0;c<6;c++) trans[key][c]/=n;}
  log('R1 GT: '+Object.keys(nbrIdx).length+' patterns. MC simulating R2...');

  // ═══ PHASE 4: MC Simulate + Blend + Submit each seed ═══
  var r2d=await(await fetch(B+'/rounds/'+R2,{credentials:'include'})).json();
  var NSIM=400; // sims per seed

  function simOne(ig){
    var g=[];for(var y=0;y<H;y++){g[y]=[];for(var x=0;x<W;x++) g[y][x]=ig[y][x];}
    for(var st=0;st<50;st++){
      var ng=[];for(var y=0;y<H;y++){ng[y]=[];for(var x=0;x<W;x++) ng[y][x]=g[y][x];}
      var ph=Math.min(4,Math.floor(st/10));
      for(var y=0;y<H;y++) for(var x=0;x<W;x++){
        var s=g[y][x];if(s===10||s===5) continue;
        var nO=0,nS=0,nF=0;
        for(var dy=-1;dy<=1;dy++) for(var dx=-1;dx<=1;dx++){
          if(!dy&&!dx) continue;var ny=y+dy,nx=x+dx;
          if(ny<0||ny>=H||nx<0||nx>=W){nO++;continue;}
          var t=g[ny][nx];if(t===10)nO++;else if(t===1||t===2)nS++;else if(t===4)nF++;
        }
        // Hierarchical lookup: detailed → coarse → ultra-coarse
        var dk=ph+'_'+s+'_'+nS+'_'+(nO>0?1:0)+'_'+(nF>2?1:0);
        var pr=TP[dk]&&TP[dk]._n>=10?TP[dk]:null;
        if(!pr){var ck='C_'+ph+'_'+s+'_'+(nS>0?1:0)+'_'+(nO>0?1:0);pr=TP[ck]&&TP[ck]._n>=5?TP[ck]:null;}
        if(!pr){var uk='U_'+ph+'_'+s;pr=TP[uk];}
        if(!pr) continue;
        var r=Math.random(),cum=0;
        for(var ns in pr){if(ns==='_n')continue;cum+=pr[ns];if(r<cum){ng[y][x]=parseInt(ns);break;}}
      }
      g=ng;
    }
    return g;
  }

  for(var seed=0;seed<SEEDS;seed++){
    log('Seed '+seed+': MC simulating ('+NSIM+' runs)... '+minsLeft()+'min left');
    var ig=r2d.initial_states[seed].grid;
    var counts=[];
    for(var y=0;y<H;y++){counts[y]=[];for(var x=0;x<W;x++) counts[y][x]=[0,0,0,0,0,0];}
    for(var sim=0;sim<NSIM;sim++){
      var final=simOne(ig);
      for(var y=0;y<H;y++) for(var x=0;x<W;x++) counts[y][x][t2c(final[y][x])]++;
      if(sim%100===99) log('Seed '+seed+': sim '+(sim+1)+'/'+NSIM);
    }

    // Build blended prediction
    var pred=[];
    for(var y=0;y<H;y++){pred[y]=[];for(var x=0;x<W;x++){
      var mc=[];for(var c=0;c<6;c++) mc[c]=counts[y][x][c]/NSIM;

      // R1 matching: try 3x3 exact, fallback to terrain type
      var h=nbrHash(ig,y,x);
      var r1p;
      if(nbrIdx[h]&&nbrIdx[h].n>=2){
        r1p=[];for(var c=0;c<6;c++) r1p[c]=nbrIdx[h].sum[c]/nbrIdx[h].n;
      } else {
        var key=ig[y][x];
        r1p=trans[key]?trans[key].slice():[0.167,0.167,0.167,0.167,0.167,0.167];
      }

      // Dynamic blend: MC weight higher for cells MC thinks are dynamic
      var mcEnt=0;for(var c=0;c<6;c++) if(mc[c]>0.001) mcEnt-=mc[c]*Math.log(mc[c]);
      var w_mc=mcEnt>0.5?0.5:0.3;
      var w_r1=1-w_mc;

      var p=[];var sum=0;
      for(var c=0;c<6;c++){p[c]=w_mc*mc[c]+w_r1*r1p[c]+0.001;sum+=p[c];}
      var maxI=0,maxV=0;
      for(var c=0;c<6;c++){p[c]=parseFloat((p[c]/sum).toFixed(6));if(p[c]>maxV){maxV=p[c];maxI=c;}}
      sum=0;for(var c=0;c<6;c++) sum+=p[c];
      p[maxI]=parseFloat((p[maxI]+(1.0-sum)).toFixed(6));
      pred[y][x]=p;
    }}

    // Submit
    log('Seed '+seed+': Submitting...');
    var resp=await fetch(B+'/submit',{method:'POST',credentials:'include',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({round_id:R2,seed_index:seed,prediction:pred})});
    var res=await resp.json();
    log('Seed '+seed+': '+res.status+' ✓');
    await new Promise(function(r){setTimeout(r,500);});
  }

  log('=== ALL 5 SEEDS SUBMITTED === '+minsLeft()+'min remaining');
  return 'Done! All seeds submitted with '+reps.length+' replays × '+NSIM+' MC sims, blended with R1 GT matching.';
})();
