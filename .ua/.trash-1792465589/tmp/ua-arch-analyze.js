const fs=require('fs');
try{
const d=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const nodes=d.fileNodes, imp=d.importEdges||[], all=d.allEdges||[];
const ids=new Set(nodes.map(n=>n.id)), byId={}; nodes.forEach(n=>byId[n.id]=n);
const paths=nodes.map(n=>n.filePath.split('/'));
let pre=[]; if(paths.length){for(let i=0;;i++){const s=paths[0][i];if(s===undefined||!paths.every(p=>p.length>i+1&&p[i]===s))break;pre.push(s);}}
const grp={}; const gOf={};
for(const n of nodes){const p=n.filePath.split('/').slice(pre.length);let g=p.length>1?p[0]:'root';
 if(g==='root'){const b=p[0]; if(/(\.test\.|\.spec\.|^test_)/.test(b))g='test';}
 (grp[g]=grp[g]||[]).push(n.id);gOf[n.id]=g;}
const types={};nodes.forEach(n=>(types[n.type]=types[n.type]||[]).push(n.id));
const fin={},fout={};const ig={},intra={};
for(const e of imp){if(!ids.has(e.source)||!ids.has(e.target))continue;fout[e.source]=(fout[e.source]||0)+1;fin[e.target]=(fin[e.target]||0)+1;
 const a=gOf[e.source],b=gOf[e.target];intra[a]=intra[a]||{internalEdges:0,totalEdges:0};intra[b]=intra[b]||{internalEdges:0,totalEdges:0};
 if(a===b){intra[a].internalEdges++;intra[a].totalEdges++;}else{intra[a].totalEdges++;intra[b].totalEdges++;const k=a+'->'+b;ig[k]=(ig[k]||0)+1;}}
Object.values(intra).forEach(v=>v.density=v.totalEdges?+(v.internalEdges/v.totalEdges).toFixed(3):0);
const cc={};for(const e of all){const s=byId[e.source],t=byId[e.target];if(!s||!t||s.type===t.type)continue;const k=[s.type,t.type,e.type].join('|');cc[k]=(cc[k]||0)+1;}
const pats={api:['routes','api','controllers','endpoints','handlers','serializers','controller','routers','blueprints'],service:['services','core','lib','domain','logic','signals','internal','composables','mailers','jobs','channels'],data:['models','db','data','persistence','repository','entities','migrations','entity','sql','database','schema'],ui:['components','views','pages','ui','layouts','screens'],middleware:['middleware','plugins','interceptors','guards'],utility:['utils','helpers','common','shared','tools','templatetags','pkg'],config:['config','constants','env','settings','management','commands'],test:['__tests__','test','tests','spec','specs'],types:['types','interfaces','schemas','contracts','dtos','dto','request','response'],hooks:['hooks'],state:['store','state','reducers','actions','slices'],assets:['assets','static','public'],entry:['cmd','bin'],documentation:['docs','documentation','wiki'],infrastructure:['deploy','deployment','infra','infrastructure','k8s','kubernetes','helm','charts','terraform','tf','docker'],'ci-cd':['.github','.gitlab','.circleci']};
const pm={};for(const g of Object.keys(grp)){for(const[l,a]of Object.entries(pats))if(a.includes(g.toLowerCase())){pm[g]=l;break;}}
const fp=nodes.map(n=>n.filePath);
const infra=fp.filter(p=>/(^|\/)(Dockerfile|docker-compose|Makefile|Jenkinsfile)|\.tf(vars)?$|\.github\/workflows|\.gitlab-ci/.test(p));
const docG=new Set(nodes.filter(n=>n.type==='document').map(n=>gOf[n.id]));
const dir=[];const seen=new Set();for(const k of Object.keys(ig)){const[a,b]=k.split('->');const pk=[a,b].sort().join('|');if(seen.has(pk))continue;seen.add(pk);const ab=ig[k],ba=ig[b+'->'+a]||0;dir.push(ab>=ba?{dependent:a,dependsOn:b}:{dependent:b,dependsOn:a});}
const out={scriptCompleted:true,commonPrefix:pre.join('/'),directoryGroups:grp,nodeTypeGroups:types,
crossCategoryEdges:Object.entries(cc).map(([k,c])=>{const[f,t,e]=k.split('|');return{fromType:f,toType:t,edgeType:e,count:c}}),
interGroupImports:Object.entries(ig).map(([k,c])=>{const[f,t]=k.split('->');return{from:f,to:t,count:c}}).sort((a,b)=>b.count-a.count),
intraGroupDensity:intra,patternMatches:pm,
deploymentTopology:{hasDockerfile:fp.some(p=>/Dockerfile/.test(p)),hasCompose:fp.some(p=>/docker-compose/.test(p)),hasK8s:fp.some(p=>/k8s|kubernetes|helm/.test(p)),hasTerraform:fp.some(p=>/\.tf$/.test(p)),hasCI:fp.some(p=>/\.github\/workflows|\.gitlab-ci|Jenkinsfile/.test(p)),infraFiles:infra},
dataPipeline:{schemaFiles:fp.filter(p=>/\.(sql|graphql|gql|proto|prisma)$/.test(p)),migrationFiles:fp.filter(p=>/migrations\//.test(p)),dataModelFiles:fp.filter(p=>/(^|\/)(models?|entities)\//.test(p)),apiHandlerFiles:fp.filter(p=>/(routes|controllers|api)\//.test(p))},
docCoverage:{groupsWithDocs:docG.size,totalGroups:Object.keys(grp).length,coverageRatio:+(docG.size/Object.keys(grp).length).toFixed(2),undocumentedGroups:Object.keys(grp).filter(g=>!docG.has(g))},
dependencyDirection:dir,fileStats:{totalFileNodes:nodes.length,filesPerGroup:Object.fromEntries(Object.entries(grp).map(([g,a])=>[g,a.length])),nodeTypeCounts:Object.fromEntries(Object.entries(types).map(([g,a])=>[g,a.length]))},
fileFanIn:fin,fileFanOut:fout};
fs.writeFileSync(process.argv[3],JSON.stringify(out,null,2));
}catch(e){console.error(e.stack);process.exit(1);}
