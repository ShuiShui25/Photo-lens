let systems = window.PHOTO_LENS_SYSTEMS;
const $ = selector => document.querySelector(selector);
let token = null, current = null, modelStage = 4, activeJob = null;
const stages = ['modeling', 'regions', 'photoz'];
let showRegionOutlines = true, showSpecz = false;
function element(tag, className, text) { const node=document.createElement(tag); if(className)node.className=className;if(text!==undefined)node.textContent=text;return node; }
function gallery() {
  $('#cards').replaceChildren();
  $('#count').textContent=systems.length;
  for(const system of systems){
    const card=element('a','card');card.href=`#system=${encodeURIComponent(system.id)}`;
    const img=element('img','card-image');img.src=system.mugshot;img.alt=system.id;card.append(img);
    const body=element('div','card-body'),tags=element('div','tags');for(const t of ['PAUS','HSC'])tags.append(element('span','tag',t));body.append(tags,element('div','card-title',system.id),element('p','',`RA ${system.ra}° · Dec ${system.dec}°`),element('p','',`z lens ${system.lens_redshift} · z source ${system.source_redshift}`));
    const foot=element('div','card-foot');foot.append(element('span','','Three-stage analysis'),element('span','','Open system →'));body.append(foot);card.append(body);$('#cards').append(card);
  }
}
function image(id,path,refresh=false){const url=path+(refresh?`?t=${Date.now()}`:'');$(`#${id}-image`).src=url;$(`#${id}-link`).href=url;}
function refresh(stage=null,bust=false){
  if(!current)return;const name=current.id;
  if(!stage||stage==='modeling')image('model',`1.lens-modeling/${name}/output/stage_previews/stage${modelStage}_fit.png`,bust);
  if(!stage||stage==='regions')image('regions',`2.region-detection/regions/${name}/${name}_regions${showRegionOutlines?'':'_no_outlines'}.png`,bust);
  if(!stage||stage==='photoz'){image('flux',`3.photometric-estimation/results/${name}/flux.png`,bust);image('pz',`3.photometric-estimation/results/${name}/pz${showSpecz?'_specz':''}.png`,bust);}
}
function route(){
  const key=new URLSearchParams(location.hash.slice(1)).get('system');current=systems.find(x=>x.id===key)||null;
  $('#gallery').hidden=!!current;$('#detail').hidden=!current;
  if(!current)return;
  $('#system-name').textContent=current.id;$('#system-subtitle').textContent=current.bands;$('#detail-mugshot').src=current.mugshot;
  $('#system-tags').replaceChildren(...[current.field,...current.aliases].map(x=>element('span','tag',x)));
  $('#facts').replaceChildren();
  for(const [label,value] of [['Coordinates',`${current.ra}°, ${current.dec}°`],['Reference redshifts',`${current.lens_redshift} / ${current.source_redshift}`],['Einstein radius',current.einstein_radius],['Reference','Lens catalogue ↗'],['Image separation',current.image_separation],['Imaging',current.imaging],['Narrow bands','40 / 40 available'],['Field',current.field]]){
    const dl=element('dl','fact'),dd=element('dd','',value);if(label==='Reference'){const a=element('a','',value);a.href=current.reference;a.target='_blank';a.rel='noopener';dd.replaceChildren(a);}dl.append(element('dt','',label),dd);$('#facts').append(dl);
  }
  for(const [index,stage] of stages.entries())$(`[data-stage="${stage}"] .notebook`).href=`${index+1}.${['lens-modeling','region-detection','photometric-estimation'][index]}/${current.id}.ipynb`;
  refresh();window.scrollTo(0,0);if(token)restoreJobs();
}
function setBusy(busy){for(const button of document.querySelectorAll('[data-run],[data-fit]'))button.disabled=busy||!token;}
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json','X-Photo-Lens-Token':token},body:JSON.stringify(body)}:{cache:'no-store'});const data=await response.json();if(!response.ok)throw Error(data.error||`HTTP ${response.status}`);return data;}
function displayJob(job){
  if(!current||job.system!==current.id)return;
  const status=$(`[data-status="${job.stage}"]`);status.dataset.state=job.state;
  const elapsed=job.started?` · ${Math.round(((job.finished||Date.now()/1000)-job.started))} s`:'';
  status.textContent=({queued:'Queued',running:'Running',succeeded:'Complete',failed:'Failed — see task output',cancelled:'Stopped'}[job.state]||job.state)+elapsed;
  const log=$(`[data-log="${job.stage}"]`);log.textContent=job.log||'Waiting for the worker…';log.scrollTop=log.scrollHeight;
  $(`[data-cancel="${job.stage}"]`).hidden=!['queued','running'].includes(job.state);
  if(job.state==='failed')log.parentElement.open=true;
}
async function poll(id){
  try{
    const job=await api(`/api/jobs/${id}`);displayJob(job);
    if(['queued','running'].includes(job.state)){activeJob=job;setBusy(true);setTimeout(()=>poll(id),1500);}
    else {activeJob=null;setBusy(false);if(job.state==='succeeded')refresh(job.stage,true);}
  }catch(error){activeJob=null;setBusy(false);$('#connection').textContent=`Connection lost: ${error.message}`;$('#offline').hidden=false;}
}
async function start(stage,mode='render'){
  if(!token||!current)return;
  setBusy(true);
  try{const job=await api('/api/jobs',{system:current.id,stage,mode});activeJob=job;displayJob(job);poll(job.id);}
  catch(error){setBusy(false);$(`[data-status="${stage}"]`).textContent=error.message;}
}
async function restoreJobs(){
  try{const jobs=(await api('/api/jobs')).filter(j=>j.system===current?.id);for(const stage of stages){const job=jobs.filter(j=>j.stage===stage).at(-1);if(job)displayJob(job);}const running=jobs.find(j=>['queued','running'].includes(j.state));if(running&&!activeJob){activeJob=running;setBusy(true);poll(running.id);}}catch{}
}
for(const [n,label] of ['SIE + MGE','Power-law mass','Two source bases','Pixelized source'].entries()){
  const button=element('button',n===3?'active':'',`${n+1} · ${label}`);button.onclick=()=>{modelStage=n+1;for(const b of $('#model-tabs').children)b.classList.toggle('active',b===button);refresh('modeling');};$('#model-tabs').append(button);
}
for(const button of document.querySelectorAll('[data-run]'))button.onclick=()=>start(button.dataset.run);
$('[data-fit]').onclick=()=>start('modeling','fit');
for(const button of document.querySelectorAll('[data-cancel]'))button.onclick=async()=>{if(activeJob)displayJob(await api(`/api/jobs/${activeJob.id}/cancel`,{}));};
$('#toggle-regions').onclick=()=>{
  showRegionOutlines=!showRegionOutlines;
  $('#toggle-regions').textContent=showRegionOutlines?'Hide region outlines':'Show region outlines';
  $('#toggle-regions').setAttribute('aria-pressed',String(showRegionOutlines));
  $('#regions-image').alt=showRegionOutlines?'Lens, source and symmetric background regions':'System image without region outlines';
  refresh('regions',true);
};
$('#toggle-specz').onclick=()=>{
  showSpecz=!showSpecz;
  $('#toggle-specz').textContent=showSpecz?'Hide lens/source spec-z':'Show lens/source spec-z';
  $('#toggle-specz').setAttribute('aria-pressed',String(showSpecz));
  $('#pz-image').alt=showSpecz?'Lens and source p(z) with spectroscopic redshift reference lines':'Lens and source photometric redshift probability distributions';
  refresh('photoz',true);
};
gallery();route();setBusy(false);window.addEventListener('hashchange',route);
(async()=>{try{token=(await api('/api/session')).token;systems=await api('/api/systems');gallery();route();$('#connection').textContent='● Local backend connected';setBusy(false);await restoreJobs();}catch{$('#connection').textContent='Static preview';$('#offline').hidden=false;setBusy(false);}})();
