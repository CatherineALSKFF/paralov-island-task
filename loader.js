/**
 * Lightweight v4 loader — sets up simulator + functions without running MC.
 * Paste into browser console, then run sims incrementally with _M.runRegime(r, nsim)
 */
(async function() {
  var BASE = 'https://api.ainm.no/astar-island';
  var FLOOR = 0.01, PRED_ALPHA = 0.15, GT_ALPHA = 0.08;
  var _t0 = Date.now();
  function MS() { return ((Date.now()-_t0)/1000).toFixed(1)+'s'; }
  window._M = {};
  var M = window._M;
  function log(m) { console.log('[M4 '+MS()+'] '+m); }

  // Fetch round data (read-only)
  log('Fetching round data...');
  var rounds = await (await fetch(BASE+'/rounds',{credentials:'include'})).json();
  var active = rounds.find(function(r){return r.status==='active';});
  var scoring = rounds.find(function(r){return r.status==='scoring';});
  var completed = rounds.filter(function(r){return r.status==='completed';}).sort(function(a,b){return b.round_number-a.round_number;});
  var ROUND = active || scoring || completed[0] || rounds[rounds.length-1];
  var ROUND_ID = ROUND.id;
  var detail = await (await fetch(BASE+'/rounds/'+ROUND_ID,{credentials:'include'})).json();
  M.d = detail; M.rid = ROUND_ID; M.rounds = rounds;
  var H = M.H = detail.map_height, W = M.W = detail.map_width, SEEDS = M.S = detail.seeds_count;
  log('Round '+detail.round_number+' ('+ROUND_ID.slice(0,8)+'), '+W+'x'+H+', '+SEEDS+' seeds, status='+ROUND.status);

  // RNG
  M.mkRng = function(seed) {
    var t = seed|0;
    return function() {
      t=(t+0x6D2B79F5)|0;
      var x=Math.imul(t^(t>>>15),1|t);
      x=(x+Math.imul(x^(x>>>7),61|x))^x;
      return ((x^(x>>>14))>>>0)/4294967296;
    };
  };

  // Settlement class
  function Settle(x,y,hp,al) {
    this.x=x;this.y=y;this.pop=1;this.food=0.5;this.wealth=0;
    this.defense=0.5;this.tech=0;this.hasPort=!!hp;this.hasLongship=false;
    this.alive=al!==false;this.ownerId=null;
  }

  M.t2c = function(t) {
    if(t===10||t===11||t===0) return 0;
    if(t===1) return 1; if(t===2) return 2; if(t===3) return 3;
    if(t===4) return 4; if(t===5) return 5; return 0;
  };

  function isCoastal(grid,x,y) {
    var dirs=[[-1,0],[1,0],[0,-1],[0,1]];
    for(var d=0;d<4;d++){var ny=y+dirs[d][0],nx=x+dirs[d][1];
      if(ny>=0&&ny<H&&nx>=0&&nx<W&&grid[ny][nx]===10)return true;}
    return false;
  }

  // v4 simulator
  M.sim = function(iGrid, iSettles, rng, P) {
    var grid = iGrid.map(function(r){return r.slice();});
    var settles = iSettles.map(function(s,i){
      var ns=new Settle(s.x,s.y,s.has_port,s.alive);ns.ownerId=i;return ns;
    });
    function shuffle(arr){for(var i=arr.length-1;i>0;i--){var j=Math.floor(rng()*(i+1));var tmp=arr[i];arr[i]=arr[j];arr[j]=tmp;}return arr;}

    for(var year=0;year<50;year++){
      var aliveList=shuffle(settles.filter(function(s){return s.alive;}));

      // GROWTH
      for(var si=0;si<aliveList.length;si++){var s=aliveList[si];
        var fr=P.foodRadius||1,fg=0;
        for(var dy=-fr;dy<=fr;dy++)for(var dx=-fr;dx<=fr;dx++){
          if(!dy&&!dx)continue;var ny=s.y+dy,nx=s.x+dx;
          if(ny<0||ny>=H||nx<0||nx>=W)continue;
          var t=grid[ny][nx];
          if(t===4)fg+=P.foodForest;else if(t===11||t===0)fg+=P.foodPlains;
        }
        s.food+=fg;
        if(s.food>P.growthTh){s.pop+=0.1*(1+s.wealth*0.05)*(1+s.tech*0.1);s.food-=P.growthTh*0.5;}
        s.defense=Math.min(s.defense+0.02*s.pop*(1+s.tech*0.05),s.pop*0.8);
        if(s.pop>1&&s.wealth>0.1)s.tech=Math.min(s.tech+(P.techGrowth||0.05)*(1+s.wealth*0.02),P.techMax||5);
        if(s.hasPort&&!s.hasLongship&&s.wealth>(P.longshipCost||0.5)&&rng()<P.longshipChance){s.hasLongship=true;s.wealth-=(P.longshipCost||0.5)*0.5;}
        if(!P.noExpand&&s.pop>=P.expandPopTh&&s.food>P.expandTh&&rng()<P.expandChance*(1+s.tech*0.05)){
          var cands=[];
          for(var ey=-P.expandDist;ey<=P.expandDist;ey++)for(var ex=-P.expandDist;ex<=P.expandDist;ex++){
            if(!ey&&!ex)continue;var eny=s.y+ey,enx=s.x+ex;
            if(eny<0||eny>=H||enx<0||enx>=W)continue;
            var et=grid[eny][enx];if(et===0||et===11||et===4){
              var close=false;for(var oi=0;oi<settles.length;oi++){var o=settles[oi];if(o.alive&&Math.abs(o.x-enx)+Math.abs(o.y-eny)<2){close=true;break;}}
              if(!close)cands.push({x:enx,y:eny});
            }
          }
          if(cands.length){var c=cands[Math.floor(rng()*cands.length)];grid[c.y][c.x]=1;
            var ns2=new Settle(c.x,c.y,false,true);ns2.pop=0.5;ns2.food=s.food*0.3;ns2.tech=s.tech*0.5;ns2.ownerId=s.ownerId;
            settles.push(ns2);s.food*=0.5;s.pop*=0.8;
          }
        }
        if(!s.hasPort&&rng()<P.portChance&&isCoastal(grid,s.x,s.y)){s.hasPort=true;grid[s.y][s.x]=2;}
      }

      // CONFLICT
      if(!P.noConflict){
        aliveList=shuffle(settles.filter(function(s){return s.alive;}));
        for(var ai=0;ai<aliveList.length;ai++){var a=aliveList[ai];if(!a.alive)continue;
          var range=a.hasLongship?P.longRaidRange:P.raidRange;
          if(rng()<(a.food<0.3?P.despRaid:P.raidChance)){
            var tgts=[];for(var ti=0;ti<aliveList.length;ti++){var tt=aliveList[ti];
              if(tt!==a&&tt.alive&&tt.ownerId!==a.ownerId&&Math.abs(tt.x-a.x)+Math.abs(tt.y-a.y)<=range)tgts.push(tt);}
            if(tgts.length){var tg=tgts[Math.floor(rng()*tgts.length)];
              var techAdv=(a.tech-tg.tech)*0.1;
              var ap=a.pop*P.raidStr*(1+a.wealth*0.05+techAdv);
              var dp=tg.pop*(1+tg.defense*0.3+tg.tech*0.05);
              if(ap>dp*(0.8+rng()*0.4)){
                var st=tg.food*P.loot;a.food+=st;tg.food-=st;
                a.wealth+=tg.wealth*P.loot*0.5;tg.wealth=Math.max(0,tg.wealth-tg.wealth*P.loot*0.5);
                tg.defense*=0.7;
                if(rng()<P.conquerChance){tg.ownerId=a.ownerId;tg.defense*=0.5;
                  if(rng()<(P.destroyOnConquest||0.15)){tg.alive=false;grid[tg.y][tg.x]=3;}}
              }else{a.defense*=0.9;}
            }
          }
        }
      }

      // TRADE
      if(!P.noTrade){
        var ports=settles.filter(function(s){return s.alive&&s.hasPort;});
        for(var pi=0;pi<ports.length;pi++)for(var pj=pi+1;pj<ports.length;pj++){
          var sp=ports[pi],pp=ports[pj];
          if(Math.abs(pp.x-sp.x)+Math.abs(pp.y-sp.y)>P.tradeRange)continue;
          var tm=sp.ownerId===pp.ownerId?1:0.5;var techM=1+(sp.tech+pp.tech)*0.05;
          sp.food+=P.tradeFood*0.5*tm*techM;pp.food+=P.tradeFood*0.5*tm*techM;
          sp.wealth+=P.tradeWealth*tm*techM;pp.wealth+=P.tradeWealth*tm*techM;
          if(sp.tech>pp.tech+0.1)pp.tech+=(sp.tech-pp.tech)*(P.techDiffusion||0.1)*tm;
          if(pp.tech>sp.tech+0.1)sp.tech+=(pp.tech-sp.tech)*(P.techDiffusion||0.1)*tm;
        }
      }

      // WINTER
      var sev=P.constWinter?P.winterBase:P.winterBase+(rng()-0.5)*P.winterVar;
      aliveList=settles.filter(function(s){return s.alive;});
      for(var wi=0;wi<aliveList.length;wi++){var ws=aliveList[wi];
        ws.food-=sev*(0.8+ws.pop*0.2);ws.pop=Math.max(0.1,ws.pop-sev*0.05);
        if(ws.food<P.collapseTh&&rng()<P.collapseChance){
          ws.alive=false;grid[ws.y][ws.x]=3;
          var nearby=settles.filter(function(n){return n.alive&&n.ownerId===ws.ownerId&&Math.abs(n.x-ws.x)+Math.abs(n.y-ws.y)<=(P.dispersalRange||4)&&n!==ws;});
          if(nearby.length){var popSh=ws.pop*(P.dispersalFraction||0.5)/nearby.length;
            for(var ni=0;ni<nearby.length;ni++){nearby[ni].pop+=popSh;}}
        }
      }

      // ENVIRONMENT
      for(var ey2=0;ey2<H;ey2++)for(var ex2=0;ex2<W;ex2++){
        if(grid[ey2][ex2]===3){
          var nearS=settles.filter(function(s){return s.alive&&Math.abs(s.x-ex2)+Math.abs(s.y-ey2)<=(P.rebuildRange||3);});
          var thriving=nearS.filter(function(s){return s.pop>1&&s.food>0.5;});
          if(thriving.length&&rng()<(P.rebuildChance||0.07)){
            if(isCoastal(grid,ex2,ey2)&&rng()<(P.portRestoreChance||0.4)){
              grid[ey2][ex2]=2;var ns3=new Settle(ex2,ey2,true,true);
              var patron=thriving[Math.floor(rng()*thriving.length)];
              ns3.pop=patron.pop*0.3;ns3.food=patron.food*0.2;ns3.tech=patron.tech*0.5;ns3.ownerId=patron.ownerId;
              settles.push(ns3);patron.pop*=0.85;patron.food*=0.7;
            }else{
              grid[ey2][ex2]=1;var ns4=new Settle(ex2,ey2,false,true);
              var patron2=thriving[Math.floor(rng()*thriving.length)];
              ns4.pop=patron2.pop*0.3;ns4.food=patron2.food*0.2;ns4.tech=patron2.tech*0.5;ns4.ownerId=patron2.ownerId;
              settles.push(ns4);patron2.pop*=0.85;patron2.food*=0.7;
            }
          }else if(rng()<(P.forestReclaim||0.06)){grid[ey2][ex2]=4;
          }else if(rng()<(P.plainsReclaim||0.03)){grid[ey2][ex2]=11;}
        }
      }
    }
    return grid;
  };

  // Parameter regimes
  var bn={techGrowth:0.05,techMax:5,techDiffusion:0.1,longshipCost:0.5,destroyOnConquest:0.15,dispersalRange:4,dispersalFraction:0.5,portRestoreChance:0.4,plainsReclaim:0.03,foodRadius:1};
  function rg(b,o){var r={};for(var k in bn)r[k]=bn[k];for(var k2 in b)r[k2]=b[k2];if(o)for(var k3 in o)r[k3]=o[k3];return r;}
  function interp(p1,p2,t){var intK=['expandDist','rebuildRange','raidRange','longRaidRange','tradeRange','dispersalRange'];
    var r={};for(var k in p1){if(typeof p1[k]==='number'){var v=p1[k]*(1-t)+p2[k]*t;r[k]=intK.indexOf(k)>=0?Math.round(v):parseFloat(v.toFixed(4));}else r[k]=p1[k];}return r;}

  var conserv=rg({foodForest:.3,foodPlains:.08,growthTh:1.8,expandTh:4,expandPopTh:2,expandChance:.04,expandDist:3,portChance:.06,longshipChance:.05,raidRange:3,longRaidRange:7,raidChance:.15,despRaid:.2,raidStr:.5,loot:.3,conquerChance:.08,tradeRange:5,tradeFood:.2,tradeWealth:.15,winterBase:1,winterVar:.6,collapseTh:-.5,collapseChance:.4,forestReclaim:.08,ruinDecay:.05,rebuildChance:.05,rebuildRange:3});
  var balanced=rg({foodForest:.4,foodPlains:.12,growthTh:1.3,expandTh:3,expandPopTh:1.5,expandChance:.08,expandDist:3,portChance:.1,longshipChance:.08,raidRange:4,longRaidRange:8,raidChance:.2,despRaid:.25,raidStr:.55,loot:.35,conquerChance:.12,tradeRange:6,tradeFood:.3,tradeWealth:.2,winterBase:.85,winterVar:.45,collapseTh:-.8,collapseChance:.3,forestReclaim:.06,ruinDecay:.03,rebuildChance:.07,rebuildRange:3});
  var modAgg=rg({foodForest:.45,foodPlains:.14,growthTh:1.1,expandTh:2.5,expandPopTh:1.3,expandChance:.1,expandDist:4,portChance:.12,longshipChance:.1,raidRange:4,longRaidRange:9,raidChance:.22,despRaid:.28,raidStr:.6,loot:.4,conquerChance:.15,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.75,winterVar:.4,collapseTh:-1,collapseChance:.25,forestReclaim:.05,ruinDecay:.025,rebuildChance:.08,rebuildRange:4});
  var aggressive=rg({foodForest:.5,foodPlains:.16,growthTh:.9,expandTh:2,expandPopTh:1,expandChance:.12,expandDist:4,portChance:.15,longshipChance:.12,raidRange:5,longRaidRange:10,raidChance:.25,despRaid:.3,raidStr:.65,loot:.45,conquerChance:.18,tradeRange:7,tradeFood:.4,tradeWealth:.3,winterBase:.65,winterVar:.35,collapseTh:-1.2,collapseChance:.2,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4});
  var ultraH=rg({foodForest:.2,foodPlains:.05,growthTh:2.5,expandTh:5,expandPopTh:3,expandChance:.02,expandDist:2,portChance:.03,longshipChance:.03,raidRange:4,longRaidRange:8,raidChance:.25,despRaid:.3,raidStr:.7,loot:.4,conquerChance:.12,tradeRange:4,tradeFood:.15,tradeWealth:.1,winterBase:1.5,winterVar:.8,collapseTh:-.3,collapseChance:.55,forestReclaim:.12,ruinDecay:.08,rebuildChance:.03,rebuildRange:2},{destroyOnConquest:.3});
  var ultraP=rg({foodForest:.5,foodPlains:.18,growthTh:1,expandTh:2,expandPopTh:1.2,expandChance:.12,expandDist:4,portChance:.12,longshipChance:.1,raidRange:2,longRaidRange:5,raidChance:.08,despRaid:.1,raidStr:.3,loot:.2,conquerChance:.05,tradeRange:6,tradeFood:.35,tradeWealth:.25,winterBase:.6,winterVar:.3,collapseTh:-1.5,collapseChance:.15,forestReclaim:.04,ruinDecay:.02,rebuildChance:.1,rebuildRange:4},{destroyOnConquest:.05});
  var hiExpLo=rg({foodForest:.45,foodPlains:.15,growthTh:1,expandTh:2,expandPopTh:1,expandChance:.15,expandDist:5,portChance:.12,longshipChance:.05,raidRange:2,longRaidRange:4,raidChance:.05,despRaid:.08,raidStr:.3,loot:.15,conquerChance:.02,tradeRange:8,tradeFood:.4,tradeWealth:.3,winterBase:.6,winterVar:.2,collapseTh:-2,collapseChance:.1,forestReclaim:.03,ruinDecay:.01,rebuildChance:.12,rebuildRange:5},{destroyOnConquest:.05});
  var mildCon=rg({foodForest:.35,foodPlains:.1,growthTh:1.5,expandTh:3.5,expandPopTh:1.8,expandChance:.06,expandDist:3,portChance:.08,longshipChance:.06,raidRange:3,longRaidRange:7,raidChance:.18,despRaid:.22,raidStr:.5,loot:.3,conquerChance:.1,tradeRange:5,tradeFood:.25,tradeWealth:.18,winterBase:.9,winterVar:.5,collapseTh:-.6,collapseChance:.35,forestReclaim:.07,ruinDecay:.04,rebuildChance:.06,rebuildRange:3});

  M.PS = [conserv, mildCon, balanced, modAgg, aggressive, ultraH, ultraP,
    rg(balanced,{noTrade:true}), rg(balanced,{noExpand:true}), rg(balanced,{noConflict:true}), rg(balanced,{constWinter:true}),
    interp(modAgg,aggressive,0.33), interp(modAgg,aggressive,0.67), hiExpLo,
    interp(conserv,balanced,0.5), interp(balanced,modAgg,0.5), interp(mildCon,balanced,0.5),
    interp(balanced,hiExpLo,0.5), interp(modAgg,hiExpLo,0.5), interp(aggressive,hiExpLo,0.5),
    rg(balanced,{foodRadius:2,foodForest:.2,foodPlains:.06}), rg(modAgg,{foodRadius:2,foodForest:.22,foodPlains:.07}), rg(aggressive,{foodRadius:2,foodForest:.25,foodPlains:.08})
  ];
  M.RN = ['Conserv','MildCon','Balance','ModAgg','Aggress','UltraH','UltraP','NoTrade','NoExpan','NoCnflt','CstWntr','MA-A33','MA-A67','HiExpLo','Con-Bal','Bal-MA','MC-Bal','Bal-HiX','MA-HiX','Agg-HiX','Bal-WR','MA-WR','Agg-WR'];
  var NR = M.PS.length;
  log(NR+' regimes defined. Simulator ready.');

  // MC storage
  M.rc = {}; M.pd = {}; M.gt = {}; M.simCounts = {};

  // Run MC for ONE regime (call incrementally)
  M.runRegime = function(r, nsim) {
    nsim = nsim || 50;
    var P = M.PS[r];
    if (!M.rc[r]) M.rc[r] = {};
    for (var s = 0; s < SEEDS; s++) {
      if (!M.rc[r][s]) {
        M.rc[r][s] = [];
        for (var y = 0; y < H; y++) { M.rc[r][s][y] = []; for (var x = 0; x < W; x++) M.rc[r][s][y][x] = {}; }
      }
      var prev = M.simCounts[r+'_'+s] || 0;
      for (var i = 0; i < nsim; i++) {
        var rng = M.mkRng(s * 100000 + r * 10000 + (prev+i) * 7 + 42);
        var fg = M.sim(detail.initial_states[s].grid, detail.initial_states[s].settlements, rng, P);
        for (var y2 = 0; y2 < H; y2++) for (var x2 = 0; x2 < W; x2++) {
          var c = M.t2c(fg[y2][x2]);
          M.rc[r][s][y2][x2][c] = (M.rc[r][s][y2][x2][c] || 0) + 1;
        }
      }
      M.simCounts[r+'_'+s] = prev + nsim;
    }
    // Build distributions
    M.pd[r] = {}; M.gt[r] = {};
    for (var s2 = 0; s2 < SEEDS; s2++) {
      M.pd[r][s2] = []; M.gt[r][s2] = [];
      for (var y3 = 0; y3 < H; y3++) {
        M.pd[r][s2][y3] = []; M.gt[r][s2][y3] = [];
        for (var x3 = 0; x3 < W; x3++) {
          var ct = M.rc[r][s2][y3][x3], p = [], g2 = [], sm = 0, sm2 = 0;
          for (var k = 0; k < 6; k++) { p[k] = (ct[k]||0)+PRED_ALPHA; sm += p[k]; g2[k] = (ct[k]||0)+GT_ALPHA; sm2 += g2[k]; }
          for (var k2 = 0; k2 < 6; k2++) { p[k2] /= sm; g2[k2] /= sm2; }
          M.pd[r][s2][y3][x3] = p; M.gt[r][s2][y3][x3] = g2;
        }
      }
    }
    log('Regime '+r+' ('+M.RN[r]+'): '+(M.simCounts[r+'_0'])+' sims/seed done');
    return M.simCounts[r+'_0'];
  };

  // Score
  M.score = function(pred, gt) {
    var totalKL=0,totalEnt=0;
    for(var y=0;y<H;y++)for(var x=0;x<W;x++){
      var g=gt[y][x],ent=0;
      for(var c=0;c<6;c++)if(g[c]>1e-6)ent-=g[c]*Math.log(g[c]);
      if(ent<0.01)continue;
      var kl=0;
      for(var c2=0;c2<6;c2++)if(g[c2]>1e-6)kl+=g[c2]*Math.log(g[c2]/Math.max(pred[y][x][c2],1e-10));
      totalKL+=Math.max(0,kl)*ent;totalEnt+=ent;
    }
    var wkl=totalEnt>0?totalKL/totalEnt:0;
    return Math.max(0,Math.min(100,100*Math.exp(-3*wkl)));
  };

  // Blend
  M.blend = function(weights, seed) {
    var totalW=0;for(var i=0;i<NR;i++)totalW+=(weights[i]||0);if(totalW===0)totalW=1;
    var pred=[];
    for(var y=0;y<H;y++){pred[y]=[];for(var x=0;x<W;x++){
      var p=[0,0,0,0,0,0];
      for(var r=0;r<NR;r++){var w=weights[r]||0;if(w===0||!M.pd[r]||!M.pd[r][seed])continue;
        var rd=M.pd[r][seed][y][x];for(var c=0;c<6;c++)p[c]+=w*rd[c];}
      for(var c2=0;c2<6;c2++)p[c2]/=totalW;
      for(var it=0;it<5;it++){var below=false;
        for(var c3=0;c3<6;c3++)if(p[c3]<FLOOR){p[c3]=FLOOR;below=true;}
        if(!below)break;var s2=0;for(var c4=0;c4<6;c4++)s2+=p[c4];var exc=s2-1;
        if(Math.abs(exc)>1e-10){var above=0;for(var c5=0;c5<6;c5++)if(p[c5]>FLOOR)above+=p[c5];
          if(above>0)for(var c6=0;c6<6;c6++)if(p[c6]>FLOOR)p[c6]-=exc*(p[c6]/above);}
      }
      for(var c7=0;c7<6;c7++)p[c7]=Math.max(FLOOR,parseFloat(p[c7].toFixed(6)));
      var s3=0;for(var c8=0;c8<6;c8++)s3+=p[c8];for(var c9=0;c9<6;c9++)p[c9]/=s3;
      var sum4=0,maxI=0,maxV=0;
      for(var c10=0;c10<6;c10++){p[c10]=parseFloat(p[c10].toFixed(6));sum4+=p[c10];if(p[c10]>maxV){maxV=p[c10];maxI=c10;}}
      p[maxI]=parseFloat((p[maxI]+(1-sum4)).toFixed(6));
      pred[y][x]=p;
    }}return pred;
  };

  // Cross-score: score regime A predictions vs regime B GT
  M.crossScore = function(predR, gtR, seed) {
    if(!M.pd[predR]||!M.gt[gtR])return -1;
    return M.score(M.pd[predR][seed], M.gt[gtR][seed]);
  };

  // Self-test: run one regime and report terrain counts
  M.test = function(r, seed) {
    seed = seed || 0; r = r || 2;
    var P = M.PS[r], init = detail.initial_states[seed];
    var rng = M.mkRng(seed*100000+r*10000+999);
    var fg = M.sim(init.grid, init.settlements, rng, P);
    var ns=0,np=0,nr2=0,nf=0,ne=0;
    for(var y=0;y<H;y++)for(var x=0;x<W;x++){var t=fg[y][x];
      if(t===1)ns++;else if(t===2)np++;else if(t===3)nr2++;else if(t===4)nf++;else if(t===0||t===11)ne++;}
    return {regime:M.RN[r],settle:ns,port:np,ruin:nr2,forest:nf,empty:ne};
  };

  // Score matrix: score each regime's prediction against every other regime's GT
  M.scoreMatrix = function(seed) {
    seed = seed || 0;
    var results = [];
    var loaded = [];
    for (var r = 0; r < NR; r++) if (M.pd[r] && M.pd[r][seed]) loaded.push(r);
    for (var i = 0; i < loaded.length; i++) {
      var predR = loaded[i];
      var worst = 999, avg = 0;
      for (var j = 0; j < loaded.length; j++) {
        var gtR = loaded[j];
        var sc = M.crossScore(predR, gtR, seed);
        if (sc < worst) worst = sc;
        avg += sc;
      }
      avg /= loaded.length;
      results.push({regime: predR, name: M.RN[predR], worst: Math.round(worst*10)/10, avg: Math.round(avg*10)/10});
    }
    results.sort(function(a,b){return b.worst-a.worst;});
    return results;
  };

  // Fetch GT (read-only, post-round)
  M.fetchGT = async function(rid) {
    rid = rid || ROUND_ID;
    M.realGT = {};
    for (var s = 0; s < SEEDS; s++) {
      var resp = await fetch(BASE+'/analysis/'+rid+'/'+s, {credentials:'include'});
      if (resp.status !== 200) { log('Seed '+s+': '+resp.status+' (not ready)'); continue; }
      var data = await resp.json();
      M.realGT[s] = data;
      log('Seed '+s+': score='+data.score+' GT loaded');
    }
    return M.realGT;
  };

  // Score vs real GT
  M.vsGT = function() {
    if (!M.realGT) { log('No GT! Run _M.fetchGT()'); return; }
    var loaded = [];
    for (var r = 0; r < NR; r++) if (M.pd[r] && M.pd[r][0]) loaded.push(r);
    var results = [];
    for (var i = 0; i < loaded.length; i++) {
      var r2 = loaded[i], totalSc = 0, seeds = [];
      for (var s = 0; s < SEEDS; s++) {
        if (!M.realGT[s] || !M.pd[r2][s]) continue;
        var sc = M.score(M.pd[r2][s], M.realGT[s].ground_truth);
        totalSc += sc; seeds.push(Math.round(sc*10)/10);
      }
      results.push({regime: r2, name: M.RN[r2], avg: Math.round(totalSc/SEEDS*10)/10, seeds: seeds});
    }
    results.sort(function(a,b){return b.avg-a.avg;});
    return results;
  };

  M.submitSeed = async function(s, preds) {
    preds = preds || M.finalPreds;
    log('*** SUBMITTING seed '+s+' ***');
    var resp = await fetch(BASE+'/submit', {method:'POST',credentials:'include',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({round_id:ROUND_ID,seed_index:s,prediction:preds[s]})});
    var r = await resp.json();
    log('Seed '+s+': '+resp.status);
    return r;
  };
  M.submitAll = async function(preds) {
    preds = preds || M.finalPreds;
    for (var s = 0; s < SEEDS; s++) { await M.submitSeed(s, preds); await new Promise(function(ok){setTimeout(ok,600);}); }
  };

  log('=== LOADER READY === Run _M.runRegime(r, nsim) to build sims incrementally');
  log('Commands: _M.runRegime(0,50), _M.test(2,0), _M.scoreMatrix(0), _M.fetchGT(), _M.vsGT()');
  return M;
})()
