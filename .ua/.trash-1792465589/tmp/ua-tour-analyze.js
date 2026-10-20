const fs=require('fs');
try{
const [inp,out]=process.argv.slice(2);
const d=JSON.parse(fs.readFileSync(inp,'utf8'));
const nodes=d.nodes,edges=d.edges,ids=new Set(nodes.map(n=>n.id));
const fin={},fout={},inS={},outS={};
nodes.forEach(n=>{inS[n.id]=new Set();outS[n.id]=new Set();});
edges.forEach(e=>{if(!ids.has(e.source)||!ids.has(e.target)||e.source===e.target)return;outS[e.source].add(e.target);inS[e.target].add(e.source);});
nodes.forEach(n=>{fin[n.id]=inS[n.id].size;fout[n.id]=outS[n.id].size;});
const byName=Object.fromEntries(nodes.map(n=>[n.id,n]));
const fanInRanking=nodes.map(n=>({id:n.id,fanIn:fin[n.id],name:n.name})).sort((a,b)=>b.fanIn-a.fanIn).slice(0,20);
const fanOutRanking=nodes.map(n=>({id:n.id,fanOut:fout[n.id],name:n.name})).sort((a,b)=>b.fanOut-a.fanOut).slice(0,20);
const fo=Object.values(fout).sort((a,b)=>b-a),fi=Object.values(fin).sort((a,b)=>a-b);
const foT=fo[Math.floor(fo.length*0.1)]||0,fiB=fi[Math.floor(fi.length*0.25)]||0;
const EP=['index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js','mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py','__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php','App.swift','Application.kt','main.cpp','main.c'];
const cands=nodes.map(n=>{let s=0;const fp=n.filePath||'';const depth=fp.split('/').length-1;
 if(n.type==='document'){if(/^readme\.(md|txt)$/i.test(fp))s+=5;else if(depth===0&&fp.endsWith('.md'))s+=2;}
 else if(n.type==='file'){if(EP.includes(n.name))s+=3;if(depth<=1)s+=1;if(fout[n.id]>=foT&&fout[n.id]>0)s+=1;if(fin[n.id]<=fiB)s+=1;}
 return {id:n.id,score:s,name:n.name,summary:n.summary};}).sort((a,b)=>b.score-a.score||fout[b.id]-fout[a.id]);
const entryPointCandidates=cands.slice(0,5);
const start=cands.find(c=>byName[c.id].type==='file').id;
const adj={};edges.forEach(e=>{if(['imports','calls'].includes(e.type)&&ids.has(e.target)){(adj[e.source]=adj[e.source]||[]).push(e.target);}});
const order=[start],depthMap={[start]:0},q=[start];
while(q.length){const c=q.shift();for(const t of adj[c]||[]){if(!(t in depthMap)){depthMap[t]=depthMap[c]+1;order.push(t);q.push(t);}}}
const byDepth={};order.forEach(i=>{(byDepth[depthMap[i]]=byDepth[depthMap[i]]||[]).push(i);});
const pick=ts=>nodes.filter(n=>ts.includes(n.type)).map(n=>({id:n.id,name:n.name,type:n.type,summary:n.summary}));
const nonCodeFiles={documentation:pick(['document']),infrastructure:pick(['service','pipeline','resource']),data:pick(['table','schema','endpoint']),config:pick(['config'])};
// clusters
const rel={};edges.forEach(e=>{if(['imports','calls'].includes(e.type))rel[e.source+'>'+e.target]=1;});
const und={};edges.forEach(e=>{if(!ids.has(e.source)||!ids.has(e.target)||e.source===e.target)return;(und[e.source]=und[e.source]||new Set()).add(e.target);(und[e.target]=und[e.target]||new Set()).add(e.source);});
const clusters=[];const seen=new Set();
Object.keys(rel).forEach(k=>{const [a,b]=k.split('>');if(rel[b+'>'+a]){const key=[a,b].sort().join('|');if(seen.has(key))return;seen.add(key);
 const cl=new Set([a,b]);for(const n of nodes){if(cl.size>=5)break;if(cl.has(n.id))continue;let c=0;cl.forEach(m=>{if(und[n.id]&&und[n.id].has(m))c++;});if(c>=2)cl.add(n.id);}
 const arr=[...cl];let ec=edges.filter(e=>cl.has(e.source)&&cl.has(e.target)).length;clusters.push({nodes:arr,edgeCount:ec});}});
clusters.sort((a,b)=>b.edgeCount-a.edgeCount);
const nodeSummaryIndex=Object.fromEntries(nodes.map(n=>[n.id,{name:n.name,type:n.type,summary:n.summary}]));
fs.writeFileSync(out,JSON.stringify({scriptCompleted:true,entryPointCandidates,fanInRanking,fanOutRanking,bfsTraversal:{startNode:start,order,depthMap,byDepth},nonCodeFiles,clusters:clusters.slice(0,10),layers:{count:d.layers.length,list:d.layers},nodeSummaryIndex,totalNodes:nodes.length,totalEdges:edges.length},null,2));
}catch(e){console.error(e.stack);process.exit(1);}
