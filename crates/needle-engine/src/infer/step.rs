//! The single-token step as one team job.
//!
//! A decode step is twenty layers of small dependent pieces. Forking and
//! joining the team for each would cost more than most pieces take, so the
//! whole stack runs as one job: members advance through each layer's phases
//! together and meet at a barrier between them. Wide pieces are split
//! (the gate projection rows, activation groups, projection rows, heads,
//! Kronecker rows, output columns); the small reductions every later piece
//! needs (norms, the gates, the MLP conditioning) are worked out by every
//! member for itself, which costs less than a barrier. Every value is
//! computed exactly as [`Model::chunk_native`] computes it for one token.

use super::*;
use crate::qlinear::{GROUP, prep_group, quant_with};
use crate::team::Barrier;

/// How many members a step runs on. A member the OS takes off its core
/// (or moves to a slower one) stalls every phase: the others sit at the
/// barrier for a scheduler quantum, many times a step. Each member adds up
/// its barrier waits over the step, and a step where some member waited
/// more than half of it counts as a stall. Recurring stalls drop a member;
/// a long stall-free stretch adds one back on trial. Results never depend
/// on the count.
pub(super) struct Members {
    state: std::sync::Mutex<MemberState>,
}

struct MemberState {
    n: usize,
    /// Steps and stalls in the current window, and steps since a stall.
    steps: u32,
    stalls: u32,
    quiet: u32,
    /// When the last step ended (cycle counter).
    last_end: u64,
    /// Wait fractions seen (`NX_MEMBERS_LOG`): under 0.1, 0.2, 0.3, 0.5,
    /// 0.7, and the rest.
    hist: [u32; 6],
}

/// Steps per stall window, and stalls in one that drop a member.
const STALL_WINDOW: u32 = 32;
const STALLS_TO_DROP: u32 = 3;
/// Stall-free steps before one more member is tried.
const QUIET_TO_GROW: u32 = 1024;
/// The wait fraction that means a member was off its core: an uneven split
/// leaves a member waiting up to a third of a step, a scheduler quantum far
/// more.
const STALL_WAIT: f64 = 0.6;
/// A gap before a step (cycle counter ticks at 24 MHz: 2 ms) after which
/// the workers have parked.
const PAUSE_TICKS: u64 = 48_000;

pub(super) fn members() -> &'static Members {
    static M: std::sync::OnceLock<Members> = std::sync::OnceLock::new();
    M.get_or_init(|| Members {
        state: std::sync::Mutex::new(MemberState { n: 0, steps: 0, stalls: 0, quiet: 0, last_end: 0, hist: [0; 6] }),
    })
}

impl Members {
    fn fixed() -> bool {
        static FIXED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FIXED.get_or_init(|| std::env::var_os("NEEDLE_THREADS").is_some())
    }

    /// The count in use.
    pub(super) fn current(&self, max: usize) -> usize {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.n == 0 || st.n > max || Self::fixed() {
            // Start one below the team: a core left to the rest of the
            // system keeps members from being preempted at all.
            st.n = if Self::fixed() { max } else { max.saturating_sub(1).max(1) };
            crate::team::set_active(st.n);
        }
        st.n
    }

    /// A step on `n` members that started at `start` (cycle counter), in
    /// which some member waited `wait` of the step's wall time at barriers.
    fn report(&self, n: usize, max: usize, start: u64, wait: f64) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        // A step after a pause finds the workers parked, and their wake-up
        // shows as waiting; it says nothing about the machine's load.
        let after_pause = start.wrapping_sub(st.last_end) > PAUSE_TICKS;
        st.last_end = ticks();
        if after_pause {
            return;
        }
        if std::env::var_os("NX_MEMBERS_LOG").is_some() {
            let bin = [0.1, 0.2, 0.3, 0.5, 0.7].iter().position(|&b| wait < b).unwrap_or(5);
            st.hist[bin] += 1;
            if st.hist.iter().sum::<u32>() == 512 {
                eprintln!("members {n}: wait fractions <.1 <.2 <.3 <.5 <.7 more = {:?}", st.hist);
                st.hist = [0; 6];
            }
        }
        if Self::fixed() || n != st.n {
            return;
        }
        let stalled = wait > STALL_WAIT;
        st.steps += 1;
        st.stalls += u32::from(stalled);
        st.quiet = if stalled { 0 } else { st.quiet + 1 };
        let lo = max.min(3);
        if st.steps >= STALL_WINDOW {
            if st.stalls >= STALLS_TO_DROP && st.n > lo {
                st.n -= 1;
                st.quiet = 0;
                crate::team::set_active(st.n);
                if std::env::var_os("NX_MEMBERS_LOG").is_some() {
                    eprintln!("members: {} stalls in {} steps, down to {}", st.stalls, st.steps, st.n);
                }
            }
            st.steps = 0;
            st.stalls = 0;
        }
        if st.quiet >= QUIET_TO_GROW && st.n < max {
            st.n += 1;
            st.quiet = 0;
            crate::team::set_active(st.n);
            if std::env::var_os("NX_MEMBERS_LOG").is_some() {
                eprintln!("members: quiet, up to {}", st.n);
            }
        }
    }
}

/// The CPU's cycle counter (its rate does not matter here, only ratios).
#[inline]
fn ticks() -> u64 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: reads the virtual counter register.
    unsafe {
        let v: u64;
        std::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags));
        v
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        static START: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(web_time::Instant::now).elapsed().as_nanos() as u64
    }
}

/// A buffer the members share; each phase writes disjoint parts.
#[derive(Clone, Copy)]
pub(super) struct Buf<T>(pub(super) SyncPtr<T>, usize);

impl<T> Buf<T> {
    pub(super) fn new(v: &mut [T]) -> Self {
        Self(SyncPtr(v.as_mut_ptr()), v.len())
    }

    /// # Safety
    /// No other member touches `r` during the phase.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn at(&self, r: std::ops::Range<usize>) -> &mut [T] {
        debug_assert!(r.end <= self.1);
        unsafe { self.0.slice(r) }
    }

    /// # Safety
    /// No member writes the buffer during the phase.
    pub(super) unsafe fn all(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.0.ptr(), self.1) }
    }
}

impl Model {
    pub(crate) fn step_native(&self, s: &mut Session, tok: u32, outputs: Outputs, want_cells: bool) -> ForwardOut {
        const NB: usize = 32;
        const HN: usize = NB * NB;
        let c = self.config();
        let q = self.q.as_ref().expect("packed weights");
        let tables = self.tables.as_ref().expect("native tables");
        let (n, d) = (c.mhc_lanes, c.d_model);
        assert!(n == 4 && c.hada_n() == HN && HADA_COND_RANK == 8, "the native path covers the shipped geometry");
        let nc = n * d;
        let l_total = c.num_layers;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        let width = qd + kd + vdim;
        let taps = c.qkv_conv_taps;
        let half = qk / 2;
        let group = nh / nkv;
        let p0 = s.tokens.len();
        assert!(p0 < c.max_seq_len, "sequence exceeds max_seq_len");
        s.tokens.push(tok);
        let attend = s.attend;
        let st = s.nat.as_mut().expect("native state");
        mark_slots(&mut st.slot_pos, p0, p0 + 1, attend);
        let sites = c.engram_layers.len();
        let mut sc = STEP.with(|cell| cell.borrow_mut().take()).filter(|sc| sc.fits(nc, sites, l_total)).unwrap_or_else(|| {
            StepScratch::new(
                nc,
                d,
                tables.phi[0].rows,
                q.qkvg[0].out,
                qd,
                nh * vd,
                sites,
                l_total,
                (q.qkvg[0].act_bits(), q.out[0].act_bits()),
            )
        });
        let StepScratch {
            slot_pos,
            xs,
            p,
            proj,
            qbuf,
            act_h,
            act_o,
            o,
            att,
            cell_rows,
            mbuf,
            tbuf,
            zbuf,
            parts,
            e_rows,
            e_acts,
            ekv,
            evals,
            ..
        } = &mut sc;
        slot_pos.clear();
        slot_pos.extend_from_slice(&st.slot_pos);
        let slot_pos: &[i64] = slot_pos;

        let scale = (d as f32).sqrt();
        let row = q.embedding.row(tok as usize);
        for lane in 0..n {
            for (o, v) in xs[lane * d..(lane + 1) * d].iter_mut().zip(&row) {
                *o = v * scale;
            }
        }
        let mut cells = want_cells.then(|| vec![0f32; (l_total + 1) * d]);
        if let Some(cl) = cells.as_mut() {
            lane_mean(xs, d, &mut cl[..d]);
        }
        // Engram n-gram rows for this position (the same slots in every
        // site's tables); the members do the rest inside the job.
        let (orders, eheads, sub) = c.engram_geometry();
        let n_tables = orders.len() * eheads;
        let erows: Vec<Option<usize>> = {
            let stride = if c.engram_seed_heads > 0 { c.engram_seed_heads } else { eheads };
            let mut v = vec![None; n_tables];
            for (oi, &order) in orders.iter().enumerate() {
                for h in 0..eheads {
                    if p0 + 1 < order {
                        continue;
                    }
                    let mut acc = ENGRAM_SEED.wrapping_mul((oi * stride + h + 1) as u32);
                    for j in 0..order {
                        let t = if p0 >= j { s.tokens[p0 - j] } else { 0 };
                        acc = (acc ^ t).wrapping_mul(ENGRAM_PRIME);
                    }
                    acc ^= acc >> 15;
                    let tbl = oi * eheads + h;
                    v[tbl] = Some(tbl * c.engram_slots + (acc % c.engram_slots as u32) as usize);
                }
            }
            v
        };
        assert!(sites == 0 || sub == GROUP, "engram tables must be one group wide");
        for es in &mut st.engram {
            es.v8.resize((p0 + 1) * d, 0);
            es.vs.resize((p0 + 1) * d / 32, 0.0);
        }
        let eraw: Vec<(SyncPtr<i8>, SyncPtr<f32>)> =
            st.engram.iter_mut().map(|es| (SyncPtr(es.v8.as_mut_ptr()), SyncPtr(es.vs.as_mut_ptr()))).collect();
        // Rows the hash leaves out stay zero; the conv accumulates from zero.
        e_rows.fill(0.0);
        evals.fill(0.0);
        let e_parts: Vec<(Buf<i8>, Buf<f32>)> = e_acts
            .iter_mut()
            .map(|a| {
                let (q, s) = a.parts_mut();
                (Buf::new(q), Buf::new(s))
            })
            .collect();
        let e_acts = &e_acts;
        let (be_rows, bekv, bevals) = (Buf::new(e_rows), Buf::new(ekv), Buf::new(evals));
        let dil = c.engram_dilation();
        let (cos, sin) = rope_rows(tables, st, p0, 1);
        let (cos, sin) = (&cos[..half], &sin[..half]);

        // Room in every layer's caches for this position; a zero history row
        // the members fill with this position's pre-tap projections.
        let slot = slot_of(p0, attend);
        let hrows = st.layers[0].hist.len() / width;
        for ls in &mut st.layers {
            ls.reserve(slot + 1, nkv, qk, vd);
            ls.hist.resize(ls.hist.len() + width, 0.0);
        }

        // Scratch the members share.

        let groups_h = act_h.groups();
        let (ah_q, ah_s) = act_h.parts_mut();
        let (ah_q, ah_s) = (Buf::new(ah_q), Buf::new(ah_s));
        let act_h: &QAct = act_h;
        let (ao_q, ao_s) = act_o.parts_mut();
        let (ao_q, ao_s) = (Buf::new(ao_q), Buf::new(ao_s));
        let act_o: &QAct = act_o;
        let (bx, bp, bproj, bq, bo) = (Buf::new(xs), Buf::new(p), Buf::new(proj), Buf::new(qbuf), Buf::new(o));
        let (bt, bz, bcells) = (Buf::new(tbuf), Buf::new(zbuf), Buf::new(if want_cells { cell_rows } else { &mut [] }));
        let _ = mbuf;
        let batt = Buf::new(att);
        let by_chunk = fast_attention(qk, vd) && GROUP.is_multiple_of(vd);
        let bparts = Buf::new(parts);
        // Raw views of each layer's cache entries for this slot (members
        // write disjoint ones) and of its history.
        let raw: Vec<LayerRaw> = st
            .layers
            .iter_mut()
            .map(|ls| LayerRaw {
                k8: SyncPtr(ls.k8.as_mut_ptr()),
                ks: SyncPtr(ls.ks.as_mut_ptr()),
                vs: SyncPtr(ls.vs.as_mut_ptr()),
                v8t: SyncPtr(ls.v8t.as_mut_ptr()),
                vcap: ls.vcap,
                hist: SyncPtr(ls.hist.as_mut_ptr()),
            })
            .collect();
        let layers: &[LayerState] = &st.layers;
        let tm = team();
        let max_members = tm.threads();
        let nt = members().current(max_members);
        let t_start = ticks();
        let max_wait = std::sync::atomic::AtomicU64::new(0);
        let barrier = Barrier::new(nt);
        let keys = |l: usize| key_ranges(p0, p0 + 1, c.layer_window(l), attend, slot_pos);

        let _job = crate::prof::Span::new("step_job");
        tm.run_n(nt, &|tid, nt| {
            let waited = std::cell::Cell::new(0u64);
            let bar = || {
                let t = ticks();
                barrier.wait(tid);
                waited.set(waited.get() + ticks().wrapping_sub(t));
            };
            // Engram: table rows and their A8, the key|value projections,
            // the int8 value history, and the dilated conv reading it back.
            // SAFETY: disjoint units per member, phases separated by
            // barriers.
            if sites > 0 {
                unsafe {
                    for unit in share(sites * n_tables, tid, nt) {
                        let (site, tbl) = (unit / n_tables, unit % n_tables);
                        let row = be_rows.at(site * d..(site + 1) * d);
                        if let Some(r) = erows[tbl] {
                            row[tbl * sub..(tbl + 1) * sub].copy_from_slice(&q.engram_tables[site].row(r));
                        }
                        let (eq, es) = e_parts[site];
                        es.at(tbl..tbl + 1)[0] = prep_group(row, tbl, eq.at(tbl * GROUP..(tbl + 1) * GROUP), q.engram_kv[site].act_bits());
                    }
                    bar();
                    for site in 0..sites {
                        let lin = &q.engram_kv[site];
                        lin.rows_into(share(lin.out, tid, nt), &e_acts[site], SyncPtr(bekv.0.ptr().add(site * 2 * d)), 2 * d);
                    }
                    bar();
                    let groups = d / 32;
                    for unit in share(sites * groups, tid, nt) {
                        let (site, g) = (unit / groups, unit % groups);
                        let (v8, vs) = eraw[site];
                        let src = &bekv.all()[site * 2 * d + d + g * 32..site * 2 * d + d + (g + 1) * 32];
                        let dst = std::slice::from_raw_parts_mut(v8.ptr().add(p0 * d + g * 32), 32);
                        *vs.ptr().add(p0 * groups + g) = quant_i8(src, dst);
                    }
                    bar();
                    let cols = share(d / 16, tid, nt);
                    let cols = cols.start * 16..cols.end * 16;
                    for site in 0..sites {
                        let (v8, vs) = eraw[site];
                        let taps_w = &self.w.engrams[site].taps;
                        let dst = bevals.at(site * d + cols.start..site * d + cols.end);
                        for j in 0..ENGRAM_CONV_TAPS {
                            let Some(pj) = p0.checked_sub(j * dil) else { break };
                            for (o, ch) in dst.iter_mut().zip(cols.clone()) {
                                let deq = *vs.ptr().add(pj * groups + ch / 32) * *v8.ptr().add(pj * d + ch) as f32;
                                *o = taps_w[j * d + ch].mul_add(deq, *o);
                            }
                        }
                    }
                    bar();
                }
            }
            let clock = std::cell::Cell::new(web_time::Instant::now());
            let lap = |name: &'static str| {
                if tid == 0 && crate::prof::enabled() {
                    let now = web_time::Instant::now();
                    crate::prof::add(name, (now - clock.get()).as_secs_f64());
                    clock.set(now);
                }
            };
            // Per-member scratch, kept by the thread across steps (a layer
            // allocates nothing).
            let mut ms = MEMBER.with(|cell| cell.borrow_mut().take()).filter(|m| m.nx.len() == nc && m.u.len() == d).unwrap_or_else(|| {
                MemberScratch {
                    nx: vec![0.0; nc],
                    nx8: vec![0; nc],
                    u: vec![0.0; d],
                    hloc: vec![0.0; d],
                    x1: vec![0.0; d],
                    ea: vec![0.0; d],
                    eb: vec![0.0; d],
                    on: vec![0.0; d],
                    x2: vec![0.0; d],
                    h2: vec![0.0; d],
                }
            });
            let MemberScratch { nx, nx8, u, hloc, x1, ea, eb, on, x2, h2 } = &mut ms;
            let (mut z, mut mt) = ([0f32; HN], [0f32; HN]);
            let mut cond = [1f32; HN];
            for l in 0..l_total {
                let lw = &self.w.layers[l];
                let l16 = &tables.layers[l];
                let lane = l % n;
                let phi = &tables.phi[l];
                let ncols = phi.rows;
                // SAFETY (whole layer): every phase writes disjoint ranges of
                // the shared buffers and cache entries, and phases are
                // separated by barriers.
                unsafe {
                    // (a) mHC read. Every member normalizes and quantizes the
                    // whole lane stream for itself; the gate rows are split.
                    let x = bx.all();
                    let r = rinv(dot16(x, x), nc);
                    for (o, &v) in nx.iter_mut().zip(x) {
                        *o = v * r;
                    }
                    let sx = crate::qlinear::quant_scale(nx);
                    quant_with(nx, 1.0 / sx, nx8);
                    for k in share(ncols, tid, nt) {
                        bp.at(k..k + 1)[0] = (sx * phi.s[k]) * idot(&phi.q[k * phi.width..(k + 1) * phi.width], nx8) as f32;
                    }
                    bar();
                    lap("s_mhc");
                    let pr = bp.all();
                    let hp: [f32; 4] = std::array::from_fn(|j| {
                        let off = if j == lane { 4.0 } else { -4.0 };
                        let z = ((pr[j] * lw.a_pre) + off) + lw.b_pre[j];
                        1.0 / (lexpf(-z) + 1.0)
                    });
                    for (o, &v) in u.iter_mut().zip(&x[..d]) {
                        *o = v * hp[0];
                    }
                    for (b, &g) in hp.iter().enumerate().skip(1) {
                        for (o, &v) in u.iter_mut().zip(&x[b * d..(b + 1) * d]) {
                            *o = v.mul_add(g, *o);
                        }
                    }

                    // (b) engram gate and (c) input norm, every member for
                    // the whole row; (d) A8 of the normed row, split by group.
                    x1.copy_from_slice(u);
                    if let Some(site) = self.site_of_layer[l] {
                        let kr = &bekv.all()[site * 2 * d..site * 2 * d + d];
                        let (rx, rkk) = (rinv(dot16(x1, x1), d), rinv(dot16(kr, kr), d));
                        for ((a, b), (&x, &k)) in ea.iter_mut().zip(eb.iter_mut()).zip(x1.iter().zip(kr)) {
                            *a = x * rx;
                            *b = k * rkk;
                        }
                        let (a, b) = (&*ea, &*eb);
                        let alpha = 1.0 / (lexpf((-dot16(a, b)) / (d as f32).sqrt()) + 1.0);
                        for (o, &v) in x1.iter_mut().zip(&bevals.all()[site * d..(site + 1) * d]) {
                            *o = v.mul_add(alpha, *o);
                        }
                    }
                    // The input norm and its quantization, split by group
                    // (a rotation per group is real work).
                    let rh = rinv(dot16(x1, x1), d);
                    for g in share(groups_h, tid, nt) {
                        for ch in g * GROUP..((g + 1) * GROUP).min(d) {
                            let t = x1[ch] * rh;
                            hloc[ch] = t.mul_add(l16.norm_in[ch].to_f32(), t);
                        }
                        ah_s.at(g..g + 1)[0] = prep_group(hloc, g, ah_q.at(g * GROUP..(g + 1) * GROUP), q.qkvg[l].act_bits());
                    }
                    bar();
                    lap("s_h8");
                    let lin = &q.qkvg[l];
                    lin.rows_into(share(lin.out, tid, nt), act_h, bproj.0, lin.out);
                    bar();
                    lap("s_qkvg");

                    // (e)-(g) per unit (q heads, k heads, v heads): conv
                    // taps, head norm and RoPE, the KV cache and the pre-tap
                    // history.
                    let lr = &raw[l];
                    let pr = bproj.all();
                    let hist = lr.hist;
                    for unit in share(nh + 2 * nkv, tid, nt) {
                        let (start, cw, cc, w, len) = if unit < nh {
                            (unit * qk, qd, unit * qk, &l16.q_taps, qk)
                        } else if unit < nh + nkv {
                            let kh = unit - nh;
                            (qd + kh * qk, kd, kh * qk, &l16.k_taps, qk)
                        } else {
                            let vh = unit - nh - nkv;
                            (qd + kd + vh * vd, vdim, vh * vd, &l16.v_taps, vd)
                        };
                        let cur = &pr[start..start + len];
                        std::slice::from_raw_parts_mut(hist.ptr().add(hrows * width + start), len).copy_from_slice(cur);
                        let mut v = [0f32; 64];
                        let v = &mut v[..len];
                        for ((o, &wv), &x) in v.iter_mut().zip(&w[cc..cc + len]).zip(cur) {
                            *o = wv.to_f32() * x;
                        }
                        for j in 1..taps {
                            let wj = &w[j * cw + cc..j * cw + cc + len];
                            if hrows >= j {
                                let r = std::slice::from_raw_parts(hist.ptr().add((hrows - j) * width + start), len);
                                for ((o, &wv), &x) in v.iter_mut().zip(wj).zip(r) {
                                    *o = wv.to_f32().mul_add(x, *o);
                                }
                            } else {
                                for (o, &wv) in v.iter_mut().zip(wj) {
                                    *o = wv.to_f32().mul_add(0.0, *o);
                                }
                            }
                        }
                        if unit < nh + nkv {
                            zcn16_inplace(v, if unit < nh { &l16.q_norm } else { &l16.k_norm });
                            for f in 0..half {
                                let (a, b) = (v[f], v[f + half]);
                                v[f] = (-sin[f]).mul_add(b, cos[f] * a);
                                v[f + half] = sin[f].mul_add(a, cos[f] * b);
                            }
                        }
                        if unit < nh {
                            bq.at(start..start + qk).copy_from_slice(v);
                        } else if unit < nh + nkv {
                            let kh = unit - nh;
                            let k8 = std::slice::from_raw_parts_mut(lr.k8.ptr().add((slot * nkv + kh) * qk), qk);
                            *lr.ks.ptr().add(slot * nkv + kh) = quant_i8(v, k8);
                        } else {
                            let vh = unit - nh - nkv;
                            let mut vq = [0i8; 64];
                            *lr.vs.ptr().add(slot * nkv + vh) = quant_i8(v, &mut vq[..vd]);
                            for (dm, &b) in vq[..vd].iter().enumerate() {
                                *lr.v8t.ptr().add((vh * vd + dm) * lr.vcap + slot) = b;
                            }
                        }
                    }
                    bar();
                    lap("s_units");

                    // (h) attention and (i) its gate per head, then (j) A8
                    // of each 128-wide group of heads.
                    let ls = &layers[l];
                    let keys = keys(l);
                    // Heads split by key chunk when the kernel allows, so
                    // the members share the work evenly; each head's chunks
                    // merge in the next phase.
                    let n_keys = keys_total(&keys);
                    let chunks = if by_chunk { key_chunks(n_keys, false, nkv).0 } else { 1 };
                    for task in share(nh * chunks, tid, nt) {
                        let (hh, ci) = (task / chunks, task % chunks);
                        let qh = &bq.all()[hh * qk..(hh + 1) * qk];
                        if by_chunk {
                            bparts.at(task..task + 1)[0] =
                                attend_chunk(ls, qh, hh / group, nkv, qk, vd, &keys, attend, false, ci).expect("chunk in range");
                        } else {
                            attend_head(ls, qh, hh / group, nkv, qk, vd, &keys, attend, false, batt.at(hh * vd..(hh + 1) * vd));
                        }
                    }
                    bar();
                    lap("s_attn");
                    for g in share(act_o.groups(), tid, nt) {
                        for hh in (g * GROUP / vd)..((g + 1) * GROUP / vd).min(nh) {
                            let out = batt.at(hh * vd..(hh + 1) * vd);
                            if by_chunk {
                                merge_parts(&bparts.all()[hh * chunks..(hh + 1) * chunks], out);
                            }
                            for (a, &gv) in out.iter_mut().zip(&pr[width + hh * vd..width + (hh + 1) * vd]) {
                                *a /= nexp(-gv) + 1.0;
                            }
                        }
                        ao_s.at(g..g + 1)[0] = prep_group(batt.all(), g, ao_q.at(g * GROUP..(g + 1) * GROUP), q.out[l].act_bits());
                    }
                    bar();
                    lap("s_a8o");
                    let lin_o = &q.out[l];
                    lin_o.rows_into(share(lin_o.out, tid, nt), act_o, bo.0, d);
                    bar();
                    lap("s_out");

                    // (k) post block. Every member finishes the residual, the
                    // pre-MLP norm and the conditioning for itself; the
                    // Kronecker passes and the SiLU are split by rows.
                    let ga = 1.0 / (lexpf(-lw.attn_gate) + 1.0);
                    zcn16(bo.all(), &l16.post_norm, on);
                    for ((x2, &on), &x1v) in x2.iter_mut().zip(on.iter()).zip(x1.iter()) {
                        *x2 = on.mul_add(ga, x1v);
                    }
                    zcn16(x2, &l16.pre_hada, h2);
                    // The MLP with two barriers: every member works out each
                    // stage's first pass (and the epilogues) for itself and
                    // splits only the second pass; the last stage's tiles go
                    // to the member that writes those columns.
                    let wk = cond_weights(h2, &l16.cond_v);
                    for (zz, (&h, &d1)) in z.iter_mut().zip(h2.iter().zip(&l16.d1)).take(d) {
                        *zz = h * d1.to_f32();
                    }
                    z[d..].fill(0.0);
                    // Each member owns row blocks of every stage's output: both
                    // passes of a stage run on them with no barrier between,
                    // the first into a private transposed product. The
                    // epilogues split by element and need the whole stage, so
                    // a barrier sits on either side of them.
                    let rows = share(NB / 8, tid, nt);
                    let tiles = rows.start * 4..rows.end * 4;
                    let jr = {
                        let r = share(HN / 4, tid, nt);
                        r.start * 4..r.end * 4
                    };
                    mix8::<true, _, _>(l16.kron[0].as_slice(), z.as_slice(), SyncPtr(mt.as_mut_ptr()), tiles.clone());
                    mix8::<false, _, _>(mt.as_slice(), l16.kron[1].as_slice(), bt.0, tiles.clone());
                    bar();
                    {
                        let t1 = bt.all();
                        let c = &mut cond[jr.clone()];
                        c.fill(1.0);
                        for (k, &w) in wk.iter().enumerate() {
                            axpy16(c, &l16.cond_u[k * HN + jr.start..k * HN + jr.end], w);
                        }
                        let zs = bz.at(jr.clone());
                        for ((zz, j), &cv) in zs.iter_mut().zip(jr.clone()).zip(c.iter()) {
                            *zz = t1[self.w.perm1[j]].mul_add(cv * l16.d2[j].to_f32(), l16.b2[j].to_f32());
                        }
                        silu_native(zs);
                    }
                    bar();
                    mix8::<true, _, _>(l16.kron[2].as_slice(), bz.all(), SyncPtr(mt.as_mut_ptr()), tiles.clone());
                    mix8::<false, _, _>(mt.as_slice(), l16.kron[3].as_slice(), bt.0, tiles.clone());
                    bar();
                    {
                        let t2 = bt.all();
                        let zs = bz.at(jr.clone());
                        for (zz, j) in zs.iter_mut().zip(jr.clone()) {
                            *zz = t2[self.w.perm2[j]] * l16.d3[j].to_f32();
                        }
                    }
                    bar();
                    mix8::<true, _, _>(l16.kron[4].as_slice(), bz.all(), SyncPtr(mt.as_mut_ptr()), tiles.clone());
                    mix8::<false, _, _>(mt.as_slice(), l16.kron[5].as_slice(), bt.0, tiles.start..tiles.end.min(d / NB / 8 * 4));
                    bar();
                    lap("s_mlp");
                    let t3 = bt.all();

                    // The post and residual gates (every member), then the
                    // mHC write over each member's tiles.
                    let hpost: [f32; 4] = std::array::from_fn(|j| {
                        let off = if j == lane { 0.0 } else { -4.0 };
                        let z = ((bp.all()[n + j] * lw.a_post) + off) + lw.b_post[j];
                        2.0 / (lexpf(-z) + 1.0)
                    });
                    let pp = bp.all();
                    let zres: [f32; 16] = std::array::from_fn(|k| pp[2 * n + k].mul_add(lw.a_res, lw.b_res[k]));
                    let hres = sinkhorn64(&zres);
                    let xp = bx.0.ptr();
                    let cols = share(d / 16, tid, nt);
                    for ch in cols.start * 16..cols.end * 16 {
                        let y = (x2[ch] + t3[ch] * l16.d4[ch].to_f32()) - u[ch];
                        let old: [f32; 4] = std::array::from_fn(|b| *xp.add(b * d + ch));
                        let mut sum = 0f32;
                        for a in 0..n {
                            let mut v = y * hpost[a];
                            for (b, &ob) in old.iter().enumerate() {
                                v = ob.mul_add(hres[a * n + b], v);
                            }
                            *xp.add(a * d + ch) = v;
                            sum = if a == 0 { v } else { sum + v };
                        }
                        if want_cells {
                            *bcells.0.ptr().add(l * d + ch) = sum * 0.25;
                        }
                    }
                    bar();
                    lap("s_final");
                }
            }
            MEMBER.with(|cell| *cell.borrow_mut() = Some(ms));
            max_wait.fetch_max(waited.get(), std::sync::atomic::Ordering::Relaxed);
        });

        drop(_job);
        let wall = ticks().wrapping_sub(t_start).max(1);
        members().report(nt, max_members, t_start, max_wait.into_inner() as f64 / wall as f64);
        let keep = taps - 1 + ROLLBACK;
        for ls in &mut st.layers {
            let rows = ls.hist.len() / width;
            if rows > 2 * keep {
                ls.hist.drain(..(rows - keep) * width);
            }
        }
        if let Some(cl) = cells.as_mut() {
            cl[d..].copy_from_slice(cell_rows);
        }
        let rows = usize::from(!matches!(outputs, Outputs::None));
        let data = if rows == 0 {
            vec![]
        } else {
            let mut mean = vec![0f32; d];
            lane_mean(xs, d, &mut mean);
            let mut h = vec![0f32; d];
            zcn(&mean, &self.w.final_norm, &mut h);
            st.last_hidden = h.clone();
            match outputs {
                Outputs::Hidden | Outputs::LastHidden => h,
                _ => crate::prof::span("step_head", || self.head_logits(&h)),
            }
        };
        if let (Some(store), Some(cl)) = (s.cells.as_mut(), cells.as_ref()) {
            store.extend_from_slice(cl);
        }
        STEP.with(|cell| *cell.borrow_mut() = Some(sc));
        ForwardOut { data, rows, cells }
    }
}

/// Buffers a step's members share, kept by the calling thread across
/// steps.
struct StepScratch {
    slot_pos: Vec<i64>,
    xs: Vec<f32>,
    p: Vec<f32>,
    proj: Vec<f32>,
    qbuf: Vec<f32>,
    act_h: QAct,
    act_o: QAct,
    o: Vec<f32>,
    att: Vec<f32>,
    cell_rows: Vec<f32>,
    mbuf: Vec<f32>,
    tbuf: Vec<f32>,
    zbuf: Vec<f32>,
    parts: Vec<Part>,
    e_rows: Vec<f32>,
    e_acts: Vec<QAct>,
    ekv: Vec<f32>,
    evals: Vec<f32>,
    sites: usize,
    layers: usize,
}

impl StepScratch {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn new(nc: usize, d: usize, ncols: usize, proj: usize, qd: usize, attd: usize, sites: usize, layers: usize, bits: (u8, u8)) -> Self {
        const HN: usize = 1024;
        Self {
            slot_pos: vec![],
            xs: vec![0.0; nc],
            p: vec![0.0; ncols],
            proj: vec![0.0; proj],
            qbuf: vec![0.0; qd],
            act_h: QAct::empty(d, bits.0),
            act_o: QAct::empty(attd, bits.1),
            o: vec![0.0; d],
            att: vec![0.0; attd],
            cell_rows: vec![0.0; layers * d],
            mbuf: vec![0.0; HN],
            tbuf: vec![0.0; HN],
            zbuf: vec![0.0; HN],
            parts: vec![(0.0, 0.0, [0.0; 64]); attd / 64 * 8],
            e_rows: vec![0.0; sites * d],
            e_acts: (0..sites).map(|_| QAct::empty(d, 2)).collect(),
            ekv: vec![0.0; sites * 2 * d],
            evals: vec![0.0; sites * d],
            sites,
            layers,
        }
    }

    fn fits(&self, nc: usize, sites: usize, layers: usize) -> bool {
        self.xs.len() == nc && self.sites == sites && self.layers == layers
    }
}

/// One member's private buffers, kept by its thread across steps.
struct MemberScratch {
    nx: Vec<f32>,
    nx8: Vec<i8>,
    u: Vec<f32>,
    hloc: Vec<f32>,
    x1: Vec<f32>,
    ea: Vec<f32>,
    eb: Vec<f32>,
    on: Vec<f32>,
    x2: Vec<f32>,
    h2: Vec<f32>,
}

thread_local! {
    static STEP: std::cell::RefCell<Option<StepScratch>> = const { std::cell::RefCell::new(None) };
    static MEMBER: std::cell::RefCell<Option<MemberScratch>> = const { std::cell::RefCell::new(None) };
}

/// Raw pointers into one layer's caches for the members of a step.
struct LayerRaw {
    k8: SyncPtr<i8>,
    ks: SyncPtr<f32>,
    vs: SyncPtr<f32>,
    v8t: SyncPtr<i8>,
    vcap: usize,
    hist: SyncPtr<f32>,
}

unsafe impl Sync for LayerRaw {}
