//! A chunk of prompt tokens as one team job.
//!
//! A prompt feed runs the same layer phases as a single-token step over a
//! few tokens at once. Forking the team for each small piece costs more
//! than the piece; instead each member owns a fixed share of the rows and
//! carries them through every row-local piece (the mHC read, the input norm
//! and its quantization, the conv taps and RoPE, the KV write, the post
//! block and the mHC write), meeting the others only where a piece needs
//! every row (the projections) or every slot (attention). The arithmetic is
//! [`Model::chunk_native`]'s, row for row.

use super::step::Buf;
use super::*;
use crate::qlinear::prep_group;
use crate::team::Barrier;

/// Raw pointers into one layer's caches for the members of a job.
struct LayerRaw {
    k8: SyncPtr<i8>,
    ks: SyncPtr<f32>,
    vs: SyncPtr<f32>,
    v8t: SyncPtr<i8>,
    vcap: usize,
    /// The conv history with room for this chunk's rows after `hrows`.
    hist: SyncPtr<f32>,
    hrows: usize,
}

unsafe impl Sync for LayerRaw {}

impl Model {
    pub(crate) fn chunk_job(&self, s: &mut Session, toks: &[u32], outputs: Outputs, want_cells: bool) -> ForwardOut {
        let c = self.config();
        let q = self.q.as_ref().expect("packed weights");
        let tables = self.tables.as_ref().expect("native tables");
        let t = toks.len();
        let (n, d) = (c.mhc_lanes, c.d_model);
        assert!(n == 4 && c.hada_n() == 1024 && HADA_COND_RANK == 8, "the native path covers the shipped geometry");
        let nc = n * d;
        let l_total = c.num_layers;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        let width = qd + kd + vdim;
        let taps = c.qkv_conv_taps;
        let half = qk / 2;
        let group = nh / nkv;
        let heads_per = GROUP / vd;
        let p0 = s.tokens.len();
        assert!(p0 + t <= c.max_seq_len, "sequence exceeds max_seq_len");
        s.tokens.extend_from_slice(toks);
        let attend = s.attend;
        let st = s.nat.as_mut().expect("native state");
        mark_slots(&mut st.slot_pos, p0, p0 + t, attend);
        let end = p0 + t;

        // Embedding, the same row in every lane.
        let scale = (d as f32).sqrt();
        let mut xs = vec![0f32; t * nc];
        for (i, &tok) in toks.iter().enumerate() {
            let row = q.embedding.row(tok as usize);
            for lane in 0..n {
                for (o, v) in xs[i * nc + lane * d..i * nc + (lane + 1) * d].iter_mut().zip(&row) {
                    *o = v * scale;
                }
            }
        }
        let depths = l_total + 1;
        let mut cells = want_cells.then(|| vec![0f32; t * depths * d]);
        if let Some(cl) = cells.as_mut() {
            for i in 0..t {
                lane_mean(&xs[i * nc..(i + 1) * nc], d, &mut cl[i * depths * d..i * depths * d + d]);
            }
        }
        let engram = crate::prof::span("engram", || self.engram_native(st, &s.tokens, p0, t));
        let engram = engram.as_ref();
        let (cos, sin) = rope_rows(tables, st, p0, t);

        // Room in every layer's caches for this chunk's slots and history
        // rows (members write those through raw pointers).
        let slot_pos = st.slot_pos.clone();
        let raw: Vec<LayerRaw> = st
            .layers
            .iter_mut()
            .map(|ls| {
                ls.reserve(slot_of(end - 1, attend) + 1, nkv, qk, vd);
                let hrows = ls.hist.len() / width;
                ls.hist.reserve(t * width);
                LayerRaw {
                    k8: SyncPtr(ls.k8.as_mut_ptr()),
                    ks: SyncPtr(ls.ks.as_mut_ptr()),
                    vs: SyncPtr(ls.vs.as_mut_ptr()),
                    v8t: SyncPtr(ls.v8t.as_mut_ptr()),
                    vcap: ls.vcap,
                    hist: SyncPtr(ls.hist.as_mut_ptr()),
                    hrows,
                }
            })
            .collect();
        let layers: &[LayerState] = &st.layers;

        // Scratch the members share.
        let ncols = tables.phi[0].rows;
        let (bits_h, bits_o) = (q.qkvg[0].act_bits(), q.out[0].act_bits());
        let mut u = vec![0f32; t * d];
        let mut p = vec![0f32; t * ncols];
        let mut x1 = vec![0f32; t * d];
        let mut act_h = QAct::empty_rows(t, d, bits_h);
        let groups_h = act_h.groups();
        let mut proj = vec![0f32; t * q.qkvg[0].out];
        let mut qkv = vec![0f32; t * width];
        let mut att = vec![0f32; t * nh * vd];
        let mut act_o = QAct::empty_rows(t, nh * vd, bits_o);
        let groups_o = act_o.groups();
        let mut o = vec![0f32; t * d];
        let (bx, bu, bp, bx1) = (Buf::new(&mut xs), Buf::new(&mut u), Buf::new(&mut p), Buf::new(&mut x1));
        let (bproj, bqkv, batt, bo) = (Buf::new(&mut proj), Buf::new(&mut qkv), Buf::new(&mut att), Buf::new(&mut o));
        let (ah_q, ah_s) = act_h.parts_mut();
        let (ah_q, ah_s) = (Buf::new(ah_q), Buf::new(ah_s));
        let (ao_q, ao_s) = act_o.parts_mut();
        let (ao_q, ao_s) = (Buf::new(ao_q), Buf::new(ao_s));
        let (act_h, act_o) = (&act_h, &act_o);
        let bcells = cells.as_mut().map(|cl| Buf::new(cl));
        let tm = team();
        let nt = tm.active();
        let barrier = Barrier::new(nt);

        let _job = crate::prof::Span::new("chunk_job");
        tm.run_n(nt, &|tid, nt| {
            let bar = || barrier.wait(tid);
            let rows = share(t, tid, nt);
            // Per-member scratch for the whole chunk.
            let mut nx = vec![0f32; nc];
            let mut q8 = vec![0i8; nc];
            let mut hloc = vec![0f32; d];
            let (mut ea, mut eb) = (vec![0f32; d], vec![0f32; d]);
            let (mut on, mut x2, mut h2) = (vec![0f32; d], vec![0f32; d], vec![0f32; d]);
            let (mut old, mut y) = (vec![0f32; nc], vec![0f32; d]);
            for l in 0..l_total {
                let lw = &self.w.layers[l];
                let l16 = &tables.layers[l];
                let lane = l % n;
                let phi = &tables.phi[l];
                let lin = &q.qkvg[l];
                let lin_o = &q.out[l];
                // SAFETY (whole layer): a member writes only its own rows,
                // its own projection rows, its own (row, head group) tasks
                // and its own KV slots, and every phase that reads another
                // member's writes follows a barrier.
                unsafe {
                    // (a) mHC read, (b) engram gate, (c) input norm and (d)
                    // its quantization, per own row.
                    for i in rows.clone() {
                        let x = &bx.all()[i * nc..(i + 1) * nc];
                        let r = rinv(dot16(x, x), nc);
                        for (o, &v) in nx.iter_mut().zip(x) {
                            *o = v * r;
                        }
                        let sx = quant_i8(&nx, &mut q8);
                        let pr = bp.at(i * ncols..(i + 1) * ncols);
                        for (k, o) in pr.iter_mut().enumerate() {
                            *o = (sx * phi.s[k]) * idot(&phi.q[k * phi.width..(k + 1) * phi.width], &q8) as f32;
                        }
                        let mut hp = [0f32; 4];
                        for (j, g) in hp.iter_mut().enumerate() {
                            let off = if j == lane { 4.0 } else { -4.0 };
                            let z = ((pr[j] * lw.a_pre) + off) + lw.b_pre[j];
                            *g = 1.0 / (lexpf(-z) + 1.0);
                        }
                        let ur = bu.at(i * d..(i + 1) * d);
                        for (o, &v) in ur.iter_mut().zip(&x[..d]) {
                            *o = v * hp[0];
                        }
                        for (b, &g) in hp.iter().enumerate().skip(1) {
                            for (o, &v) in ur.iter_mut().zip(&x[b * d..(b + 1) * d]) {
                                *o = v.mul_add(g, *o);
                            }
                        }
                        let xr = bx1.at(i * d..(i + 1) * d);
                        xr.copy_from_slice(ur);
                        if let (Some(site), Some((keys, vals))) = (self.site_of_layer[l], engram) {
                            let kr = &keys[site][i * d..(i + 1) * d];
                            let (rx, rk) = (rinv(dot16(xr, xr), d), rinv(dot16(kr, kr), d));
                            for ((a, b), (&x, &k)) in ea.iter_mut().zip(eb.iter_mut()).zip(xr.iter().zip(kr)) {
                                *a = x * rx;
                                *b = k * rk;
                            }
                            let alpha = 1.0 / (lexpf((-dot16(&ea, &eb)) / (d as f32).sqrt()) + 1.0);
                            for (o, &v) in xr.iter_mut().zip(&vals[site][i * d..(i + 1) * d]) {
                                *o = v.mul_add(alpha, *o);
                            }
                        }
                        zcn16(xr, &l16.norm_in, &mut hloc);
                        for g in 0..groups_h {
                            let unit = i * groups_h + g;
                            ah_s.at(unit..unit + 1)[0] = prep_group(&hloc, g, ah_q.at(unit * GROUP..(unit + 1) * GROUP), bits_h);
                        }
                    }
                    bar();
                    // q | k | v | gate projection, weight rows split.
                    lin.rows_into(share(lin.out, tid, nt), act_h, bproj.0, lin.out);
                    bar();

                    // (e) conv taps, (f) head norm and RoPE, the history row
                    // and (g) the KV cache, per own row.
                    let lr = &raw[l];
                    let hrows = lr.hrows;
                    for i in rows.clone() {
                        let prev = |j: usize| -> Option<&[f32]> {
                            if j <= i {
                                Some(&bproj.all()[(i - j) * lin.out..(i - j) * lin.out + width])
                            } else if hrows + i >= j {
                                Some(std::slice::from_raw_parts(lr.hist.ptr().add((hrows + i - j) * width), width))
                            } else {
                                None
                            }
                        };
                        let out = bqkv.at(i * width..(i + 1) * width);
                        for (start, cw, w) in [(0, qd, &l16.q_taps), (qd, kd, &l16.k_taps), (qd + kd, vdim, &l16.v_taps)] {
                            let o = &mut out[start..start + cw];
                            let cur = &prev(0).expect("current row")[start..start + cw];
                            for ((o, &wv), &x) in o.iter_mut().zip(&w[..cw]).zip(cur) {
                                *o = wv.to_f32() * x;
                            }
                            for j in 1..taps {
                                let wj = &w[j * cw..(j + 1) * cw];
                                match prev(j) {
                                    Some(r) => {
                                        for ((o, &wv), &x) in o.iter_mut().zip(wj).zip(&r[start..start + cw]) {
                                            *o = wv.to_f32().mul_add(x, *o);
                                        }
                                    }
                                    None => {
                                        for (o, &wv) in o.iter_mut().zip(wj) {
                                            *o = wv.to_f32().mul_add(0.0, *o);
                                        }
                                    }
                                }
                            }
                        }
                        let (cs, sn) = (&cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
                        for hh in 0..nh + nkv {
                            let (off, scale) = if hh < nh { (hh * qk, &l16.q_norm) } else { (qd + (hh - nh) * qk, &l16.k_norm) };
                            let x = &mut out[off..off + qk];
                            zcn16_inplace(x, scale);
                            for f in 0..half {
                                let (a, b) = (x[f], x[f + half]);
                                x[f] = (-sn[f]).mul_add(b, cs[f] * a);
                                x[f + half] = sn[f].mul_add(a, cs[f] * b);
                            }
                        }
                        std::slice::from_raw_parts_mut(lr.hist.ptr().add((hrows + i) * width), width)
                            .copy_from_slice(&bproj.all()[i * lin.out..i * lin.out + width]);
                        let pos = slot_of(p0 + i, attend);
                        let row: &[f32] = out;
                        let mut vq = [0i8; 256];
                        for kvh in 0..nkv {
                            let k8 = std::slice::from_raw_parts_mut(lr.k8.ptr().add((pos * nkv + kvh) * qk), qk);
                            *lr.ks.ptr().add(pos * nkv + kvh) = quant_i8(&row[qd + kvh * qk..qd + (kvh + 1) * qk], k8);
                            *lr.vs.ptr().add(pos * nkv + kvh) = quant_i8(&row[qd + kd + kvh * vd..qd + kd + (kvh + 1) * vd], &mut vq[..vd]);
                            for (dm, &v) in vq[..vd].iter().enumerate() {
                                *lr.v8t.ptr().add((kvh * vd + dm) * lr.vcap + pos) = v;
                            }
                        }
                    }
                    bar();

                    // (h) attention and (i) its gate per (row, head group),
                    // then (j) the group's quantization.
                    let ls = &layers[l];
                    let window = c.layer_window(l);
                    for task in share(t * groups_o, tid, nt) {
                        let (i, g) = (task / groups_o, task % groups_o);
                        let pos = p0 + i;
                        let keys = key_ranges(pos, end, window, attend, &slot_pos);
                        let qrow = &bqkv.all()[i * width..(i + 1) * width];
                        for hh in g * heads_per..((g + 1) * heads_per).min(nh) {
                            let out = batt.at((i * nh + hh) * vd..(i * nh + hh + 1) * vd);
                            attend_head(ls, &qrow[hh * qk..(hh + 1) * qk], hh / group, nkv, qk, vd, &keys, attend, true, out);
                            let gate = &bproj.all()[i * lin.out + width + hh * vd..i * lin.out + width + (hh + 1) * vd];
                            for (a, &gv) in out.iter_mut().zip(gate) {
                                *a /= nexp(-gv) + 1.0;
                            }
                        }
                        let unit = i * groups_o + g;
                        ao_s.at(unit..unit + 1)[0] =
                            prep_group(&batt.all()[i * nh * vd..(i + 1) * nh * vd], g, ao_q.at(unit * GROUP..(unit + 1) * GROUP), bits_o);
                    }
                    bar();
                    lin_o.rows_into(share(lin_o.out, tid, nt), act_o, bo.0, d);
                    bar();

                    // (k) post block, per own row: residual, pre-MLP norm, the
                    // MLP, the post and residual gates, Sinkhorn, and the mHC
                    // write. The next layer's read touches only these rows,
                    // so no barrier follows.
                    let ga = 1.0 / (lexpf(-lw.attn_gate) + 1.0);
                    for i in rows.clone() {
                        zcn16(&bo.all()[i * d..(i + 1) * d], &l16.post_norm, &mut on);
                        for ((x2, &on), &x1v) in x2.iter_mut().zip(&on).zip(&bx1.all()[i * d..(i + 1) * d]) {
                            *x2 = on.mul_add(ga, x1v);
                        }
                        zcn16(&x2, &l16.pre_hada, &mut h2);
                        let mlp = self.mlp_native(l16, &h2);
                        let pr = &bp.all()[i * ncols..(i + 1) * ncols];
                        let hpost: [f32; 4] = std::array::from_fn(|j| {
                            let off = if j == lane { 0.0 } else { -4.0 };
                            let z = ((pr[n + j] * lw.a_post) + off) + lw.b_post[j];
                            2.0 / (lexpf(-z) + 1.0)
                        });
                        let zres: [f32; 16] = std::array::from_fn(|k| pr[2 * n + k].mul_add(lw.a_res, lw.b_res[k]));
                        let hres = sinkhorn64(&zres);
                        let x = bx.at(i * nc..(i + 1) * nc);
                        old.copy_from_slice(x);
                        let ur = &bu.all()[i * d..(i + 1) * d];
                        for (ch, yv) in y.iter_mut().enumerate() {
                            *yv = (x2[ch] + mlp[ch]) - ur[ch];
                        }
                        for a in 0..n {
                            let xa = &mut x[a * d..(a + 1) * d];
                            for (o, &yv) in xa.iter_mut().zip(&y) {
                                *o = yv * hpost[a];
                            }
                            for b in 0..n {
                                let h = hres[a * n + b];
                                for (o, &v) in xa.iter_mut().zip(&old[b * d..(b + 1) * d]) {
                                    *o = v.mul_add(h, *o);
                                }
                            }
                        }
                        if let Some(bc) = bcells {
                            lane_mean(x, d, bc.at((i * depths + l + 1) * d..(i * depths + l + 2) * d));
                        }
                    }
                }
            }
        });
        drop(_job);

        // The history rows the members wrote, then the usual trim.
        let keep = taps - 1 + ROLLBACK;
        for ls in &mut st.layers {
            // SAFETY: the job filled `t` rows past the old length inside the
            // capacity reserved above.
            unsafe { ls.hist.set_len(ls.hist.len() + t * width) };
            let rows = ls.hist.len() / width;
            if rows > 2 * keep {
                ls.hist.drain(..(rows - keep) * width);
            }
        }

        // Lane mean, final norm, tied head.
        let rows: Vec<usize> = match outputs {
            Outputs::None => vec![],
            Outputs::LastLogits | Outputs::LastHidden => vec![t - 1],
            Outputs::AllLogits | Outputs::Hidden => (0..t).collect(),
        };
        let mut h = vec![0f32; rows.len() * d];
        for (r, &i) in rows.iter().enumerate() {
            let mut mean = vec![0f32; d];
            lane_mean(&xs[i * nc..(i + 1) * nc], d, &mut mean);
            zcn(&mean, &self.w.final_norm, &mut h[r * d..(r + 1) * d]);
        }
        if let Some(r) = rows.len().checked_sub(1) {
            st.last_hidden = h[r * d..(r + 1) * d].to_vec();
        }
        let data = match outputs {
            Outputs::None => vec![],
            Outputs::Hidden | Outputs::LastHidden => h,
            _ => {
                crate::prof::span("head", || rows.iter().enumerate().flat_map(|(r, _)| self.head_logits(&h[r * d..(r + 1) * d])).collect())
            }
        };
        if let (Some(store), Some(cl)) = (s.cells.as_mut(), cells.as_ref()) {
            store.extend_from_slice(cl);
        }
        ForwardOut { data, rows: rows.len(), cells }
    }
}
