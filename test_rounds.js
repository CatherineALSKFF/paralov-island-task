const {computeScore, getFeatureKey} = require('./eval_harness');
const H = 40, W = 40;
const {mergeBuckets} = require('./strategies/shared');
const buckets = require('./data/gt_model_buckets.json');
const fs = require('fs');

const ROUND_IDS = {7:'36e581f1', 12:'795bfb1f'};

function loadGrid(rn, si) {
  const inits = JSON.parse(fs.readFileSync('data/inits_R'+rn+'.json', 'utf8'));
  const item = inits[si];
  const grid = [];
  for (let y = 0; y < H; y++) {
    const r = item[y] || item[String(y)];
    grid.push(Array.isArray(r) ? [...r] : Object.values(r).map(Number));
  }
  return grid;
}

function predictScore(testRound, trainRounds) {
  const model = mergeBuckets(buckets, trainRounds);
  const prefix = ROUND_IDS[testRound];
  let totalScore = 0, nSeeds = 0;
  for (let si = 0; si < 5; si++) {
    const gtFile = 'data/gt_' + prefix + '_s' + si + '.json';
    if (!fs.existsSync(gtFile)) continue;
    const gtRaw = JSON.parse(fs.readFileSync(gtFile, 'utf8'));
    const gt = gtRaw.ground_truth || gtRaw.gt;
    if (!gt || !gt[0]) { console.log('  skipping seed', si, '- no GT'); continue; }
    const grid = loadGrid(testRound, si);
    const settPos = new Set();
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        if (grid[y][x] === 1 || grid[y][x] === 2) settPos.add(y * W + x);
    console.log('  seed', si, 'grid:', grid.length, grid[0]?.length, 'cell00:', grid[0]?.[0]);
    const pred = [];
    try {
      for (let y = 0; y < H; y++) {
        const row = [];
        for (let x = 0; x < W; x++) {
          const key = getFeatureKey(grid, settPos, y, x);
          const p = model[key] ? [...model[key]] : [1/6,1/6,1/6,1/6,1/6,1/6];
          const floored = p.map(v => Math.max(v, 0.0001));
          const s = floored.reduce((a, b) => a + b, 0);
          row.push(floored.map(v => v / s));
        }
        pred.push(row);
      }
    } catch(e) {
      console.log('  ERROR at building pred:', e.message);
    }
    console.log('  pred size', pred.length, 'x', pred[0] ? pred[0].length : '??');
    totalScore += computeScore(pred, gt);
    nSeeds++;
  }
  return totalScore / nSeeds;
}

console.log('\nR12 predicted by:');
console.log('  R7 only:', predictScore(12, [7]).toFixed(1));
console.log('  R1 only:', predictScore(12, [1]).toFixed(1));
console.log('  R5 only:', predictScore(12, [5]).toFixed(1));
console.log('  R9 only:', predictScore(12, [9]).toFixed(1));
console.log('  R4 only:', predictScore(12, [4]).toFixed(1));
console.log('  All excl R12:', predictScore(12, [1,2,3,4,5,6,7,8,9,10,11,13,14]).toFixed(1));

console.log('\nR7 predicted by:');
console.log('  R12 only:', predictScore(7, [12]).toFixed(1));
console.log('  R1 only:', predictScore(7, [1]).toFixed(1));
console.log('  R5 only:', predictScore(7, [5]).toFixed(1));
console.log('  R9 only:', predictScore(7, [9]).toFixed(1));
console.log('  All excl R7:', predictScore(7, [1,2,3,4,5,6,8,9,10,11,12,13,14]).toFixed(1));
