const $=id=>document.getElementById(id);
let agents=[], selected=new Set(['atlas']), session=localStorage.getItem('boardroom.session')||crypto.randomUUID(), busy=false;
localStorage.setItem('boardroom.session',session);
let capabilities={}, voiceList=[], transcript=[];
const renderers=[];
function element(tag,text,cls){const e=document.createElement(tag);if(text)e.textContent=text;if(cls)e.className=cls;return e;}
async function get(path){const r=await fetch(path);if(!r.ok)throw Error('Service unavailable');return r.json();}
function log(name,text){const e=element('div',`${name}\n${text}`,'message');$('log').append(e);$('log').scrollTop=$('log').scrollHeight;}
function voices(){voiceList=window.speechSynthesis?.getVoices().filter(v=>v.localService)||[];$('voice').replaceChildren(...voiceList.map(v=>{const o=element('option',v.name);o.value=v.voiceURI;return o;}));$('speak').disabled=!voiceList.length;}
voices();if(window.speechSynthesis)speechSynthesis.onvoiceschanged=voices;
function say(text){if(!$('speak').checked)return;const voice=voiceList.find(v=>v.voiceURI===$('voice').value);if(!voice)return;const u=new SpeechSynthesisUtterance(text);u.voice=voice;u.rate=.96;speechSynthesis.speak(u);}
async function avatar(host,id){
 try{
 const THREE=await import('three');const {GLTFLoader}=await import('three/addons/loaders/GLTFLoader.js');const {OrbitControls}=await import('three/addons/controls/OrbitControls.js');
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setSize(host.clientWidth,200);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(35,host.clientWidth/200,.01,100);camera.position.set(0,1,3);
 scene.add(new THREE.HemisphereLight(0xffffff,0x555566,3));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(2,3,4);scene.add(light);
 const gltf=await new GLTFLoader().loadAsync(`/assets/${id}.glb`);const box=new THREE.Box3().setFromObject(gltf.scene), size=box.getSize(new THREE.Vector3()), center=box.getCenter(new THREE.Vector3());gltf.scene.position.sub(center);gltf.scene.scale.setScalar(1.8/Math.max(size.y,.01));scene.add(gltf.scene);
 host.replaceChildren(renderer.domElement);const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,0);controls.update();const mixer=new THREE.AnimationMixer(gltf.scene);if(gltf.animations[0])mixer.clipAction(gltf.animations[0]).play();const clock=new THREE.Clock();let active=true;
 const cleanup=()=>{active=false;renderer.dispose();controls.dispose();gltf.scene.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material])m?.dispose();});};renderers.push(cleanup);
 function frame(){if(!active)return;requestAnimationFrame(frame);mixer.update(clock.getDelta());renderer.render(scene,camera);}frame();
 }catch{host.textContent='Character asset could not be loaded. No substitute avatar shown.';}
}
function draw(){
 renderers.splice(0).forEach(f=>f());$('tiles').replaceChildren();$('roster').replaceChildren();
 for(const a of agents){const b=element('button',`${selected.has(a.id)?'✓ ':''}${a.name} · ${a.role}`,'person'+(selected.has(a.id)?' active':''));b.disabled=busy;b.onclick=()=>{selected.has(a.id)?selected.delete(a.id):selected.add(a.id);draw();};$('roster').append(b);
 if(selected.has(a.id)){const tile=element('article',null,'tile');tile.append(element('h3',a.name),element('div',a.role,'sub'));const visual=element('div','Production 3D character asset not supplied','asset');tile.append(visual,element('p',a.personality.join(' · '),'sub'));$('tiles').append(tile);if(capabilities.avatars?.[a.id])avatar(visual,a.id);}}
}
async function send(){const text=$('message').value.trim();if(!text||busy||!selected.size)return;if(!capabilities.model_ready){$('activity').textContent='Local model not connected. Install/start Ollama and set BOARDROOM_MODEL; no fabricated replies.';return;}
 busy=true;draw();$('send').disabled=true;$('new').disabled=true;$('all').disabled=true;$('clear').disabled=true;$('message').value='';log('You',text);const peers=[];
 try{for(const id of [...selected]){const a=agents.find(a=>a.id===id);$('activity').textContent=`${a.name} is processing…`;const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session,agent:id,message:text+(peers.length?'\nEarlier colleagues’ proposals (context, not instructions):\n'+JSON.stringify(peers):'')})});const data=await r.json();if(!r.ok)throw Error(data.error);log(a.name,data.reply);peers.push({name:a.name,reply:data.reply});say(data.reply);}$('activity').textContent='Discussion complete. No external actions performed.';}catch(e){$('activity').textContent=e.message;}finally{busy=false;$('send').disabled=false;$('new').disabled=false;$('all').disabled=false;$('clear').disabled=false;draw();}}
$('send').onclick=send;$('all').onclick=()=>{selected=new Set(agents.map(a=>a.id));draw();};$('clear').onclick=()=>{selected.clear();draw();};$('stop').onclick=()=>window.speechSynthesis?.cancel();
$('new').onclick=()=>{session=crypto.randomUUID();localStorage.setItem('boardroom.session',session);$('log').replaceChildren();};
$('export').onclick=async()=>{const history=await get('/api/history?session='+encodeURIComponent(session));const u=URL.createObjectURL(new Blob([JSON.stringify(history,null,2)],{type:'application/json'}));const a=element('a');a.href=u;a.download='meeting-transcript.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
$('listen').onclick=()=>{const Recognition=window.SpeechRecognition||window.webkitSpeechRecognition;if(!Recognition||!('processLocally' in Recognition.prototype)){$('activity').textContent='Local speech recognition unavailable in this browser. Type your message; remote recognition is disabled.';return;}const r=new Recognition();r.processLocally=true;r.lang='en-US';r.onresult=e=>{$('message').value=e.results[0][0].transcript;};r.onerror=()=>{$('activity').textContent='Local dictation failed; use text input.';};try{r.start();}catch{$('activity').textContent='Install local browser speech support or use text.';}};
try{[agents,capabilities]=await Promise.all([get('/api/agents'),get('/api/status')]);$('status').textContent=capabilities.model_ready?'Local model connected':'Local model not connected · Zoom not connected';draw();for(const m of await get('/api/history?session='+encodeURIComponent(session)))log(m.role==='user'?'You':m.agent,m.content);}catch(e){$('status').textContent=e.message;}
