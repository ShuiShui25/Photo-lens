/* Structured selection diagnostics; all file-derived content uses textContent. */
window.renderRegionDetails = function (host, data) {
  const make = (tag, cls, value) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined) node.textContent = value;
    return node;
  };
  host.replaceChildren();
  const details = make('details', 'selection-details');
  details.append(make('summary', '', 'Selection details'));
  const container = make('div');
  details.append(container);
  host.append(details);
  const fmt = (value, digits = 2) => typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits) : (value === null || value === undefined ? '—' : String(value));
  const int = value => fmt(value, 0);
  const percent = value => typeof value === 'number' ? `${fmt(100 * value, 1)}%` : '—';
  const pair = (a, b) => `${int(a)} → ${int(b)}`;
  const fate = value => {
    const labels = {anchor: 'Protected anchor', rejected: 'Rejected', skipped: 'Skipped', merged: 'Merged',
      selected: 'Selected', passed_not_selected: 'Passed; lower rank', discarded: 'Discarded'};
    const node = make('span', 'decision', labels[value] || value);
    node.dataset.fate = value;
    return node;
  };
  const section = title => {
    const detail = make('section', 'region-detail');
    detail.append(make('h3', '', title));
    container.append(detail);
    return detail;
  };
  const table = (parent, title, headings, rows) => {
    const wrap = make('div', 'diagnostic-table-scroll');
    wrap.tabIndex = 0;
    wrap.setAttribute('role', 'region');
    wrap.setAttribute('aria-label', title);
    const grid = make('table', 'diagnostic-table');
    const thead = make('thead'), head = make('tr');
    for (const heading of headings) { const th = make('th', '', heading); th.scope = 'col'; head.append(th); }
    thead.append(head); grid.append(thead);
    const body = make('tbody');
    for (const values of rows) {
      const tr = make('tr');
      for (const value of values) {
        const cell = make('td');
        if (value instanceof Node) cell.append(value); else cell.textContent = value ?? '—';
        tr.append(cell);
      }
      body.append(tr);
    }
    grid.append(body); wrap.append(grid); parent.append(wrap);
  };
  const selected = data.selected || {}, anchors = data.anchors || {};
  const contact = data.lens_contact || {}, source = data.source_selection || {};
  const satellites = data.satellites || {}, background = data.background || {};
  container.replaceChildren();
  container.append(make('h3', '', 'Selection summary'));
  container.append(make('p', 'hint',
    `${fmt(source.detect_sigma, 1)}σ source detection; one selection pass. ${data.centre_definition}`));
  const metrics = make('div', 'region-metrics');
  for (const [label, value] of [
    ['Model stage', int(data.model_stage)],
    ['Protected anchors / components', `${int(anchors.protected_anchor_count)} / ${anchors.components?.length ?? 0}`],
    ['Lens area · initial → final (px)', pair(data.lens_initial_area_pix, data.lens_area_pix)],
    ['Source area · before → after carving (px)', pair(selected.area_before_carving_pix, selected.area_pix)],
    ['Selected source', selected.selected_candidate || '—'],
    ['Integrated / effective S/N', `${fmt(selected.integrated_snr, 1)} / ${fmt(selected.effective_snr, 1)}`],
    ['S/N retained after carving', percent(selected.snr_retention)],
    ['Satellite apertures / modeled', `${int(satellites.selected_satellite_count ?? 0)} / ${int(satellites.satellite_count_in_manifest ?? 0)}`],
  ]) {
    const dl = make('dl'); dl.append(make('dt', '', label), make('dd', '', value)); metrics.append(dl);
  }
  container.append(metrics);
  if (data.warnings?.length || data.error) {
    const box = make('div', 'notice selection-warnings');
    box.append(make('strong', '', data.error ? 'Selection failed' : 'Selection notes'));
    const list = make('ul');
    for (const warning of [data.error, ...(data.warnings || [])].filter(Boolean)) list.append(make('li', '', warning));
    box.append(list); container.append(box);
  }
  const figures = section('Components, anchor positions and final apertures');
  figures.append(make('p', 'hint', `${data.source_image}. Dashed magenta: initial lens boundary; solid: final aperture. Click the figure to enlarge.`));
  figures.append(make('p', 'hint', 'Stars and component numbers mark protected source anchors; crosses mark rejected peaks. The blue cross marks the fitted lens-light centre. Coordinates are detector (x, y), starting at 0.'));
  const link = make('a'); link.href = data.figure_url; link.target = '_blank'; link.rel = 'noopener';
  const img = make('img', 'result-image'); img.src = link.href;
  img.alt = 'Four panels: detected components and anchors, initial and final lens, selected source, point-reflected background';
  link.append(img); figures.append(link);
  const anchorDetails = section('Anchor decisions and lens boundary');
  anchorDetails.append(make('p', 'hint',
    `Core radius: ${fmt(anchors.core_radius_pix)} px (0.5 PSF FWHM); minimum anchor component area: ${int(anchors.anchor_min_component_area_pix)} px; fitted Einstein radius: ${fmt(anchors.fitted_einstein_radius_pix)} px.`));
  anchorDetails.append(make('p', '',
    `Lens contact: nearest anchor ${fmt(contact.anchor_min_distance_pix)} px; required clearance ${fmt(contact.contact_distance_pix)} px (${fmt(contact.contact_distance_fwhm, 1)} FWHM). Unconstrained θE envelope: ${int(contact.unconstrained_einstein_area_pix)} px. Smallest-envelope fallback: ${contact.fallback_smallest_envelope ? 'Yes' : 'No'}. Lens radial scale: ${fmt(data.lens_scale)}.`));
  table(anchorDetails, 'Anchor decisions',
    ['Component', 'Area (px)', 'Peak S/N', 'Integrated S/N', 'Peak (x, y)', 'r peak (px)', 'r / θE', 'Contains lens centre', 'Bounding box (px)', 'Decision', 'Reason'],
    (anchors.components || []).map(c => [int(c.component_index_1based), int(c.area_pix), fmt(c.peak_residual_snr, 1),
      fmt(c.component_integrated_snr, 1), `(${int(c.peak_full_x)}, ${int(c.peak_full_y)})`, fmt(c.peak_radius_pix),
      fmt(c.peak_radius_over_einstein_radius), c.contains_lens_centre_pixel ? 'Yes' : 'No',
      c.bbox ? `x ${c.bbox.full_xmin}–${c.bbox.full_xmax}; y ${c.bbox.full_ymin}–${c.bbox.full_ymax}` : '—', fate(c.fate), c.reason]));
  const candidates = section('Source candidates: overlap, carving, gates and ranking');
  const gates = source.hard_gates || {};
  candidates.append(make('p', 'hint',
    `Overlap assignment: r_s ≥ ${fmt(source.rs_threshold, 1)}. Gates: area ≥ ${int(gates.minimum_area_pix)} px, integrated S/N ≥ ${fmt(gates.minimum_integrated_snr, 1)}, carving S/N retention ≥ ${percent(gates.minimum_carving_snr_retention)}; exclude the lens centre. Passing candidates are ranked by effective S/N (lens-light systematic fraction ε = ${fmt(source.effective_snr_epsilon, 3)}). Scroll horizontally for all decisions.`));
  table(candidates, 'Source candidate decisions',
    ['Candidate', 'Grown (px)', 'Lens overlap (px)', 'r_s reliable / accepted', 'Before carving (px)', 'Removed: direct / reflected / self', 'Pieces',
      'Final (px)', 'S/N before', 'S/N after', 'S/N retained', 'Effective S/N', 'Lens / source flux', 'Holes (px)', 'Decision', 'Reason'],
    (source.candidates || []).map(c => [c.candidate, int(c.grown_area_pix), int(c.lens_overlap_pix),
      `${int(c.rs_reliable_overlap_pix)} / ${int(c.rs_accepted_overlap_pix)}`, int(c.area_before_carving_pix),
      `${int(c.carved_direct_pix)} / ${int(c.carved_mirror_on_others_or_satellite_pix)} / ${int(c.carved_self_overlap_pix)}`,
      int(c.pieces_after_carving), int(c.area_after_carving_pix), fmt(c.integrated_snr_before_carving, 1),
      fmt(c.integrated_snr_after_carving, 1), percent(c.snr_retention), fmt(c.effective_snr, 1),
      fmt(c.lens_model_flux_fraction), int(c.hole_pixel_count), fate(c.fate), (c.reasons || []).join('; ') || '—']));
  const checks = section('Background checks and satellite selection');
  checks.append(make('p', '',
    `Background area: ${int(background.background_area_pix_full_detector)} px. Overlap with source / satellites / other source components: ${int(background.source_background_overlap_pix)} / ${int(background.background_satellite_overlap_pix)} / ${int(background.background_secondary_overlap_pix)} px.`));
  checks.append(make('p', 'hint',
    'Source and background follow exact pixel-cell boundaries. The background is reflected around the fitted lens-light centre, including its subpixel position.'));
  if (satellites.satellites?.length) {
    table(checks, 'Satellite aperture decisions', ['Satellite', 'Selected', 'Area (px)', 'Peak model S/N', 'Half-light radius (px)', 'Retained 1σ flux', 'Reason'],
      satellites.satellites.map(c => [int(c.index + 1), c.selected ? 'Yes' : 'No', int(c.area_pix),
        fmt(c.peak_model_snr), fmt(c.half_light_radius_pix), percent(c.current_1sigma_model_flux_retention), c.reason || 'Compact model aperture']));
  } else checks.append(make('p', 'hint', 'No modeled satellite aperture.'));
};
