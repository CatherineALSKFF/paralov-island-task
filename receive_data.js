const http = require('http');
const fs = require('fs');
let chunks = [];
const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(200); res.end(); return; }
  if (req.method === 'POST') {
    let body = '';
    req.on('data', d => body += d);
    req.on('end', () => {
      fs.writeFileSync('round1_data.json', body);
      console.log('Saved ' + body.length + ' bytes to round1_data.json');
      res.writeHead(200); res.end('OK');
      server.close();
      process.exit(0);
    });
  }
});
server.listen(8766, () => console.log('Receiver listening on :8766'));
