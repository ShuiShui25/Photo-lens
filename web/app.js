let systems = window.PHOTO_LENS_SYSTEMS;
const $ = selector => document.querySelector(selector);
let token = null, current = null, modelStage = 4, activeJob = null;
let submitting = false;
const stages = ['modeling', 'regions', 'photoz'];
let showRegionOutlines = true, showSpecz = false, viewedSystem = null;
let regionDetailsRequest = 0;
const fallbackVersions = new Map();
function assetUrl(path, refresh=false){
  let revision=current?.asset_revisions?.[path];
  // Static previews and an older backend still need a stable cache-busting URL.
  // Retain it across unrelated task updates; never revert to an unversioned URL.
  if(!revision){
    if(refresh||!fallbackVersions.has(path))fallbackVersions.set(path,String(Date.now()));
    revision=fallbackVersions.get(path);
  }
  const url=new URL(path,document.baseURI);
  url.searchParams.set('v',revision);
  return url.href;
}
function element(tag, className, text) { const node=document.createElement(tag); if(className)node.className=className;if(text!==undefined)node.textContent=text;return node; }
function gallery() {
  $('#cards').replaceChildren();
  $('#count').textContent=systems.length;
  for(const system of systems){
    const card=element('a','card');card.href=`#system=${encodeURIComponent(system.id)}`;
    const img=element('img','card-image');img.src=system.mugshot;img.alt=system.id;card.append(img);
    const body=element('div','card-body'),tags=element('div','tags');for(const t of ['PAUS',system.bb_survey||'HSC'])tags.append(element('span','tag',t));body.append(tags,element('div','card-title',system.id),element('p','',`RA ${system.ra}° · Dec ${system.dec}°`),element('p','',`z lens ${system.lens_redshift??'—'} · z source ${system.source_redshift??'—'}`));
    const foot=element('div','card-foot');foot.append(element('span','','Three-stage analysis'),element('span','','Open system →'));body.append(foot);card.append(body);$('#cards').append(card);
  }
}
function image(id,path,refresh=false,exists=true){
  const img=$(`#${id}-image`),link=$(`#${id}-link`);
  let empty=$(`#${id}-empty`);
  if(!empty){empty=element('p','hint','No result yet. Run the corresponding stage first.');empty.id=`${id}-empty`;link.after(empty);}
  link.hidden=!exists;empty.hidden=exists;
  if(!exists){img.removeAttribute('src');link.removeAttribute('href');return;}
  img.onerror=()=>{link.hidden=true;empty.hidden=false;};
  img.onload=()=>{link.hidden=false;empty.hidden=true;};
  const url=assetUrl(path,refresh);
  if(img.src!==url)img.src=url;
  link.href=url;
}
function refresh(stage=null,bust=false){
  if(!current)return;const name=current.id;
  if(!stage||stage==='modeling')image('model',`1.lens-modeling/${name}/output/stage_previews/stage${modelStage}_fit.png`,bust,current.has_results?.modeling??true);
  if(!stage||stage==='regions'){
    image('regions',`2.region-detection/regions/${name}/${name}_regions${showRegionOutlines?'':'_no_outlines'}.png`,bust,current.has_results?.regions??true);
    refreshRegionDetails(bust);
  }
  if(!stage||stage==='photoz'){image('flux',`3.photometric-estimation/results/${name}/flux.png`,bust,current.has_results?.photoz??true);image('pz',`3.photometric-estimation/results/${name}/pz${showSpecz?'_specz':''}.png`,bust,current.has_results?.photoz??true);}
}
async function refreshRegionDetails(bust=false){
  const request=++regionDetailsRequest, container=$('#region-diagnostics');
  $('#regions-method-status').hidden=!(current?.has_results?.regions&&!current.regions_current);
  if(!current?.region_diagnostics){container.hidden=true;container.replaceChildren();delete container.dataset.key;return;}
  const name=current.id,revision=current.region_revision,key=`${name}:${revision}`;
  if(container.dataset.key===key&&!bust)return;
  if(container.dataset.system!==name){container.replaceChildren();container.hidden=true;}
  try{
    const response=await fetch(assetUrl(current.region_diagnostics,bust),{cache:'no-store'});
    if(!response.ok)throw Error(`HTTP ${response.status}`);
    const data=await response.json();
    if(request!==regionDetailsRequest||current?.id!==name)return;
    data.figure_url=assetUrl(data.figure,bust);
    window.renderRegionDetails(container,data);
    container.dataset.key=key;container.dataset.system=name;container.hidden=false;
  }catch(error){
    if(request!==regionDetailsRequest||current?.id!==name)return;
    container.replaceChildren(element('p','hint',`Selection diagnostics could not be loaded: ${error.message}`));
    container.hidden=false;delete container.dataset.key;
  }
}
function route(){
  const key=new URLSearchParams(location.hash.slice(1)).get('system');current=systems.find(x=>x.id===key)||null;
  $('#gallery').hidden=!!current;$('#detail').hidden=!current;
  if(!current)return;
  if(viewedSystem!==current.id)modelStage=current.model_stage||4;
  for(const [index,button] of [...$('#model-tabs').children].entries()){
    button.disabled=index+1>(current.model_stage||4);
    button.classList.toggle('active',index+1===modelStage);
  }
  $('#system-name').textContent=current.id;$('#system-subtitle').textContent=current.bands;$('#detail-mugshot').src=current.mugshot;
  $('#system-tags').replaceChildren(...[current.field,...current.aliases].map(x=>element('span','tag',x)));
  $('#facts').replaceChildren();
  for(const [label,value] of [['Coordinates',`${current.ra}°, ${current.dec}°`],['Reference redshifts',`${current.lens_redshift??'—'} / ${current.source_redshift??'—'}`],['Einstein radius',current.einstein_radius],['Reference','Lens catalogue ↗'],['Image separation',current.image_separation],['Imaging',current.imaging],['Narrow bands',`${current.nb_count??40} / 40 available`],['Field',current.field]]){
    const dl=element('dl','fact'),dd=element('dd','',value);if(label==='Reference'){const a=element('a','',value);a.href=current.reference;a.target='_blank';a.rel='noopener';dd.replaceChildren(a);}dl.append(element('dt','',label),dd);$('#facts').append(dl);
  }
  for(const [index,stage] of stages.entries()){
    const notebook=$(`[data-stage="${stage}"] .notebook`);
    notebook.href=`${index+1}.${['lens-modeling','region-detection','photometric-estimation'][index]}/${current.id}.ipynb`;
    notebook.hidden=current.notebooks?.[stage]===false;
    if(viewedSystem!==current.id){
      const status=$(`[data-status="${stage}"]`);delete status.dataset.state;
      status.textContent=current.has_results?.[stage]?'Saved results':'Not run';
      $(`[data-log="${stage}"]`).textContent='Ready.';$(`[data-cancel="${stage}"]`).hidden=true;
    }
    const saved=$(`[data-saved="${stage}"]`),updated=current.result_updated_at?.[stage];
    saved.hidden=!current.has_results?.[stage];
    saved.textContent=updated
      ? `Saved results · ${new Date(updated*1000).toLocaleString()} · Shown until this stage completes a new run.`
      : 'Previously saved results · Shown until this stage completes a new run.';
  }
  $('[data-run="modeling"]').textContent=current.has_model===false?'Fit model & plot':'Rebuild model & plot';
  $('[data-fit]').hidden=current.has_model===false;
  $('[data-stage="modeling"] .hint:not(.saved-results)').textContent=current.has_model===false
    ? `No saved model. The first run fits ${current.model_stage||4} stages and can take substantially longer.`
    : `Final model: stage ${current.model_stage||4}. Rebuild evaluates the saved maximum-likelihood model. Refit runs the parameter search and may take substantially longer.`;
  $('#photoz-input-status').hidden=!current.photoz_stale;
  $('#input-status').textContent=(current.input_issues||[]).join(' ');
  $('#input-status').hidden=!(current.input_issues||[]).length;
  refresh();setBusy(!!activeJob);
  if(viewedSystem!==current.id)window.scrollTo(0,0);
  viewedSystem=current.id;if(token)restoreJobs();
}
function setBusy(busy){
  for(const button of document.querySelectorAll('[data-run],[data-fit]')){
    const stage=button.dataset.run||'modeling';
    button.disabled=busy||submitting||!token||current?.can_run?.[stage]===false;
    button.title=current?.can_run?.[stage]===false?(stage==='modeling'?'Complete the modeling inputs first.':'Complete the preceding stage and required inputs first.'):'';
  }
}
async function renewSession(){
  const session=await api('/api/session');
  token=session.token;
  $('#connection').textContent='● Local backend connected';$('#offline').hidden=true;
}
async function api(path,body){
  // A restarted backend has a new session token, even in an already open tab.
  if(body!==undefined)await renewSession();
  for(let attempt=0;attempt<2;attempt++){
    const response=await fetch(path,body!==undefined?{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json','X-Photo-Lens-Token':token},body:JSON.stringify(body)}:{cache:'no-store'});
    const data=await response.json();
    if(response.ok)return data;
    const expired=response.status===403&&(data.code==='session_expired'||data.error==='Open the local dashboard before starting a task');
    // Retry only an explicitly rejected session request, never an ambiguous
    // network failure: a submitted fit must not accidentally run twice.
    if(body!==undefined&&expired&&attempt===0){await renewSession();continue;}
    const error=Error(data.error||`HTTP ${response.status}`);error.status=response.status;throw error;
  }
}
function displayJob(job){
  if(!current||job.system!==current.id)return;
  const status=$(`[data-status="${job.stage}"]`);status.dataset.state=job.state;
  const elapsed=job.started?` · ${Math.round(((job.finished||Date.now()/1000)-job.started))} s`:'';
  status.textContent=({queued:'Queued',running:'Running',succeeded:'Complete',failed:'Failed — see task output',cancelled:'Stopped'}[job.state]||job.state)+elapsed;
  const log=$(`[data-log="${job.stage}"]`);log.textContent=job.log||'Waiting for the worker…';log.scrollTop=log.scrollHeight;
  $(`[data-cancel="${job.stage}"]`).hidden=!['queued','running'].includes(job.state);
  if(['queued','running','failed'].includes(job.state))log.parentElement.open=true;
}
async function poll(id){
  try{
    const job=await api(`/api/jobs/${id}`);displayJob(job);
    if(['queued','running'].includes(job.state)){activeJob=job;setBusy(true);setTimeout(()=>poll(id),1500);}
    else {activeJob=null;setBusy(false);if(job.state==='succeeded'){await syncSystems(true);if(current?.id===job.system)refresh(job.stage,true);}}
  }catch(error){activeJob=null;setBusy(false);$('#connection').textContent=`Connection lost: ${error.message}`;$('#offline').hidden=false;}
}
async function start(stage,mode='render'){
  if(!token||!current||submitting||activeJob)return;
  const name=current.id;
  if(stage==='modeling'&&current.has_model===false)mode='fit';
  submitting=true;
  setBusy(true);
  const status=$(`[data-status="${stage}"]`),log=$(`[data-log="${stage}"]`);
  status.dataset.state='submitting';status.textContent='Submitting task…';
  log.textContent=`Connecting to the backend and submitting ${stage} for ${name}…`;
  log.parentElement.open=true;
  try{const job=await api('/api/jobs',{system:name,stage,mode});activeJob=job;displayJob(job);poll(job.id);}
  catch(error){
    if(current?.id===name){
      status.dataset.state='failed';
      status.textContent=error.status?'Task not started — see task output':'Submission unconfirmed — see task output';
      log.textContent=error.status?`The backend rejected this task (HTTP ${error.status}).\n${error.message}`
        :`Could not confirm task submission.\n${error.message}\nCheck the backend connection and task status before trying again.`;
      log.parentElement.open=true;
    }
    if(!error.status)await restoreJobs();
  }finally{submitting=false;setBusy(!!activeJob);}
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
  $('#regions-image').alt=showRegionOutlines?'Lens, source, symmetric background and available satellite regions':'System image without region outlines';
  refresh('regions',true);
};
$('#toggle-specz').onclick=()=>{
  showSpecz=!showSpecz;
  $('#toggle-specz').textContent=showSpecz?'Hide lens/source spec-z':'Show lens/source spec-z';
  $('#toggle-specz').setAttribute('aria-pressed',String(showSpecz));
  $('#pz-image').alt=showSpecz?'Lens p(z) above and source p(z) below, each with its spectroscopic redshift reference line':'Lens p(z) in the upper panel and source p(z) in the lower panel';
  refresh('photoz',true);
};
let systemsRequest=null;
async function syncSystems(fresh=false){
  if(!token)return;
  if(systemsRequest){await systemsRequest;if(!fresh)return;}
  const request=(async()=>{
    const updated=await api('/api/systems');
    if(JSON.stringify(updated)!==JSON.stringify(systems)){systems=updated;gallery();route();}
  })();
  systemsRequest=request;
  try{
    await request;
  }finally{if(systemsRequest===request)systemsRequest=null;}
}
$('#refresh-systems').onclick=()=>syncSystems().catch(error=>{$('#connection').textContent=error.message;});
setInterval(()=>{if(!document.hidden)syncSystems().catch(()=>{});},10000);
window.addEventListener('focus',()=>syncSystems().catch(()=>{}));
gallery();route();setBusy(false);window.addEventListener('hashchange',route);
(async()=>{try{token=(await api('/api/session')).token;systems=await api('/api/systems');gallery();route();$('#connection').textContent='● Local backend connected';setBusy(false);await restoreJobs();}catch{$('#connection').textContent='Static preview';$('#offline').hidden=false;setBusy(false);}})();
