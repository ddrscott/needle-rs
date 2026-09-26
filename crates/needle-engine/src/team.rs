//! A small spin-synchronised worker team for decode.
//!
//! Decoding one token is a chain of 10-50 microsecond pieces of work; a
//! general work-stealing pool spends that long just waking threads. The team
//! keeps a few threads spinning on an epoch counter between jobs (parking
//! only after a stretch of idleness), so a fork-join costs about a
//! microsecond.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};
use std::time::Duration;
use web_time::Instant;

type Job = dyn Fn(usize, usize) + Sync;

struct Inner {
    epoch: AtomicUsize,
    remaining: AtomicUsize,
    /// Members taking part in the current job (the rest skip it).
    members: AtomicUsize,
    job: UnsafeCell<*const Job>,
    threads: usize,
    workers: OnceLock<Vec<Thread>>,
    sleeping: AtomicUsize,
}

// SAFETY: `job` is written only by the single caller of `run` (serialised by
// `lock`) before the epoch bump that publishes it, and read by workers only
// after observing that bump.
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

pub struct Team {
    inner: Arc<Inner>,
    lock: std::sync::Mutex<()>,
}

const SPIN_BEFORE_PARK: Duration = Duration::from_micros(100);
/// How long an in-job wait spins before it starts yielding.
const STALL: Duration = Duration::from_millis(1);

impl Team {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let inner = Arc::new(Inner {
            epoch: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            members: AtomicUsize::new(threads),
            job: UnsafeCell::new(std::ptr::null::<fn(usize, usize)>() as *const Job),
            threads,
            workers: OnceLock::new(),
            sleeping: AtomicUsize::new(0),
        });
        let mut handles = Vec::new();
        for tid in 1..threads {
            let inner = inner.clone();
            let h = thread::Builder::new().name(format!("needle-team-{tid}")).spawn(move || worker(inner, tid)).expect("spawn team worker");
            handles.push(h.thread().clone());
        }
        let _ = inner.workers.set(handles);
        Self { inner, lock: std::sync::Mutex::new(()) }
    }

    /// The team's size (a job may use fewer members, see [`Team::run_n`]).
    pub fn threads(&self) -> usize {
        self.inner.threads
    }

    /// Run `f(tid, n)` on the current number of members.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        self.run_n(self.active(), f)
    }

    /// How many members jobs use now: the decode step's choice
    /// ([`set_active`]), every thread until it has made one.
    pub fn active(&self) -> usize {
        match ACTIVE.load(Ordering::Relaxed) {
            0 => self.threads(),
            n => n.min(self.threads()),
        }
    }

    /// Run `f(tid, n)` on `n` members (the caller is member 0) and return
    /// when all have finished. Callers that size shared state (a barrier)
    /// by [`Team::threads`] pass that same `n`.
    pub fn run_n(&self, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let n = n.clamp(1, self.inner.threads);
        if n == 1 {
            f(0, 1);
            return;
        }
        let _g = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        self.inner.members.store(n, Ordering::Relaxed);
        // SAFETY: we block below until every worker has finished with `f`,
        // so erasing its lifetime never lets a worker see a dangling job.
        let ptr: *const Job = unsafe { std::mem::transmute::<*const (dyn Fn(usize, usize) + Sync + '_), *const Job>(f) };
        unsafe { *self.inner.job.get() = ptr };
        self.inner.remaining.store(n - 1, Ordering::Relaxed);
        self.inner.epoch.fetch_add(1, Ordering::SeqCst);
        if self.inner.sleeping.load(Ordering::SeqCst) > 0 {
            // Only the members this job uses; a worker left out stays parked.
            for t in self.inner.workers.get().into_iter().flatten().take(n - 1) {
                t.unpark();
            }
        }
        f(0, n);
        spin_until(|| self.inner.remaining.load(Ordering::Acquire) == 0);
    }
}

fn worker(inner: Arc<Inner>, tid: usize) {
    crate::cpu::prefer_performance_cores();
    let mut seen = 0usize;
    // Left out of the last job: go straight to sleep rather than spin on a
    // core other programs need (the team wakes us when it grows).
    let mut left_out = false;
    loop {
        let start = Instant::now();
        let mut spins = 0u32;
        let epoch = loop {
            let e = inner.epoch.load(Ordering::Acquire);
            if e != seen {
                break e;
            }
            spins = spins.wrapping_add(1);
            if left_out || (spins.is_multiple_of(1024) && start.elapsed() > SPIN_BEFORE_PARK) {
                // SeqCst pairs with the dispatcher's epoch bump then
                // sleeping check, so one side always sees the other.
                inner.sleeping.fetch_add(1, Ordering::SeqCst);
                if inner.epoch.load(Ordering::SeqCst) == seen {
                    thread::park_timeout(Duration::from_millis(50));
                }
                inner.sleeping.fetch_sub(1, Ordering::AcqRel);
            } else {
                std::hint::spin_loop();
            }
        };
        seen = epoch;
        let members = inner.members.load(Ordering::Relaxed);
        left_out = tid >= members;
        if left_out {
            continue;
        }
        // SAFETY: the job was published before the epoch bump we observed.
        let job = unsafe { &**inner.job.get() };
        job(tid, members);
        inner.remaining.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The member count the decode step has settled on (0: none yet).
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Record the decode step's member count for every other job.
pub fn set_active(n: usize) {
    ACTIVE.store(n, Ordering::Relaxed);
}

/// The process-wide team (`NEEDLE_THREADS`, default every performance
/// core; the calling thread is one of the members). Decode steps choose how
/// many of them to use.
pub fn team() -> &'static Team {
    static TEAM: OnceLock<Team> = OnceLock::new();
    TEAM.get_or_init(|| {
        let n = std::env::var("NEEDLE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(crate::cpu::performance_cores);
        Team::new(n)
    })
}

/// A cache line of its own.
#[repr(align(128))]
struct Padded<T>(T);

/// A spinning barrier for team members inside one `run` (a whole decode
/// step is one job; its phases meet here instead of returning to the
/// caller). Flat combining: each member publishes its arrival on its own
/// cache line and member 0 gathers them, then releases everyone through
/// one line the others only read. No member writes a line another is
/// spinning on until the release, so arrivals do not contend.
pub struct Barrier {
    slots: Box<[Padded<AtomicUsize>]>,
    release: Padded<AtomicUsize>,
}

impl Barrier {
    pub fn new(n: usize) -> Self {
        Self { slots: (0..n).map(|_| Padded(AtomicUsize::new(0))).collect(), release: Padded(AtomicUsize::new(0)) }
    }

    /// Every member calls this the same number of times, as `tid`.
    #[inline]
    pub fn wait(&self, tid: usize) {
        let n = self.slots.len();
        if n == 1 {
            return;
        }
        let epoch = self.slots[tid].0.load(Ordering::Relaxed) + 1;
        if tid == 0 {
            self.slots[0].0.store(epoch, Ordering::Relaxed);
            for s in &self.slots[1..] {
                spin_until(|| s.0.load(Ordering::Acquire) == epoch);
            }
            self.release.0.store(epoch, Ordering::Release);
        } else {
            self.slots[tid].0.store(epoch, Ordering::Release);
            spin_until(|| self.release.0.load(Ordering::Acquire) == epoch);
        }
    }
}

/// Phases of one team job: each member runs its fixed share of a phase's
/// items, then meets the others at a [`Barrier`]. The item list, not the
/// team size, fixes what each item computes, so results do not depend on
/// how many members run.
pub struct Phases {
    barrier: Barrier,
}

impl Phases {
    /// Phases for a job on `nt` members.
    pub fn new(nt: usize) -> Self {
        Self { barrier: Barrier::new(nt) }
    }

    /// Run member `tid`'s share of the phase's `items` through `f`, then
    /// wait for the rest, so every write is visible.
    #[inline]
    pub fn run(&self, tid: usize, nt: usize, items: usize, f: impl FnMut(usize)) {
        share(items, tid, nt).for_each(f);
        self.barrier.wait(tid);
    }
}

/// Spin until `done`, yielding the core after a few thousand tries: a
/// phase boundary normally clears in well under a microsecond, and when it
/// does not, a member has likely been preempted and needs the core more
/// than we do.
#[inline]
fn spin_until(done: impl Fn() -> bool) {
    // Spin on the CPU's pause hint. `yield_now` on macOS demotes the thread,
    // which under load leaves members off-core long after the barrier
    // releases, so it's only a safety valve after a millisecond.
    let mut spins = 0u32;
    let mut start = None;
    while !done() {
        spins = spins.wrapping_add(1);
        std::hint::spin_loop();
        if spins.is_multiple_of(4096) && start.get_or_insert_with(Instant::now).elapsed() > STALL {
            thread::yield_now();
        }
    }
}

/// Split `0..n` into the member's contiguous share.
#[inline]
pub fn share(n: usize, tid: usize, threads: usize) -> std::ops::Range<usize> {
    let per = n.div_ceil(threads);
    let lo = (tid * per).min(n);
    lo..((tid + 1) * per).min(n)
}

/// A raw pointer that may cross threads; members write disjoint ranges.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// The raw pointer (a method, so closures capture the `Sync` wrapper).
    #[inline]
    pub fn ptr(&self) -> *mut T {
        self.0
    }

    /// # Safety
    /// The caller guarantees `range` is in bounds and no other member
    /// touches it concurrently.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, range: std::ops::Range<usize>) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(range.start), range.end - range.start) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_member_runs_its_share() {
        let t = Team::new(4);
        let mut out = vec![0usize; 1000];
        for round in 0..100 {
            let p = SyncPtr(out.as_mut_ptr());
            t.run(&|tid, n| {
                let r = share(1000, tid, n);
                for (i, v) in unsafe { p.slice(r.clone()) }.iter_mut().enumerate() {
                    *v = r.start + i + round;
                }
            });
            assert!(out.iter().enumerate().all(|(i, &v)| v == i + round));
        }
    }
}

#[cfg(test)]
mod bench {
    #[test]
    #[ignore]
    fn bench_fork_join() {
        let t = super::team();
        for _ in 0..1000 {
            t.run(&|_, _| {});
        }
        let n = 100_000;
        let s = web_time::Instant::now();
        for _ in 0..n {
            t.run(&|_, _| {});
        }
        eprintln!("{} threads: {:.2}us per empty run", t.threads(), s.elapsed().as_secs_f64() / n as f64 * 1e6);
    }

    /// `cargo test --release -p needle-engine bench_barrier -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_barrier() {
        let t = super::team();
        let n = 20_000;
        let fb = super::Barrier::new(t.threads());
        let spin = || {
            t.run(&|tid, _| {
                for _ in 0..n {
                    fb.wait(tid);
                }
            })
        };
        spin();
        let start = web_time::Instant::now();
        spin();
        eprintln!("{} members: {:.0} ns per barrier", t.threads(), start.elapsed().as_secs_f64() / n as f64 * 1e9);
    }
}

#[cfg(test)]
mod bench_barrier {
    use super::*;

    /// `cargo test --release -p needle-engine team::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_barrier() {
        let tm = team();
        for nt in [2, 4, tm.threads()] {
            let b = Barrier::new(nt);
            let n = 20000;
            tm.run_n(nt, &|tid, _| b.wait(tid));
            let t = web_time::Instant::now();
            tm.run_n(nt, &|tid, _| {
                for _ in 0..n {
                    b.wait(tid);
                }
            });
            eprintln!("{nt} members: {:.0} ns per barrier", t.elapsed().as_secs_f64() / n as f64 * 1e9);
            let t = web_time::Instant::now();
            for _ in 0..2000 {
                tm.run_n(nt, &|_, _| {});
            }
            eprintln!("{nt} members: {:.0} ns per empty run", t.elapsed().as_secs_f64() / 2000.0 * 1e9);
        }
    }
}
