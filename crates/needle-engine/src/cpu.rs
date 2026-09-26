//! Thread sizing: work goes to the performance cores only. On Apple
//! silicon the efficiency cores are several times slower, and a fork-join
//! waits for its slowest member.

use std::sync::Once;

/// Performance cores (`hw.perflevel0.physicalcpu` on macOS), else all.
pub fn performance_cores() -> usize {
    // A browser gives a wasm module one thread.
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    #[cfg(target_os = "macos")]
    {
        let mut v: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        let name = c"hw.perflevel0.physicalcpu";
        // SAFETY: sysctlbyname writes at most `len` bytes into `v`.
        let rc = unsafe { sysctlbyname(name.as_ptr(), (&mut v as *mut i32).cast(), &mut len, std::ptr::null_mut(), 0) };
        if rc == 0 && v > 0 {
            return v as usize;
        }
    }
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Ask the scheduler to keep the calling thread on performance cores
/// (macOS QoS "user interactive"), returning the class it had so the caller
/// can restore it. A decode step waits for its slowest member, and a member
/// moved to an efficiency core is several times slower.
pub fn prefer_performance_cores() -> Option<u32> {
    #[cfg(target_os = "macos")]
    // SAFETY: both calls only read or set the calling thread's QoS class.
    unsafe {
        let mut class: u32 = 0;
        let mut rel: i32 = 0;
        let had = (pthread_get_qos_class_np(pthread_self(), &mut class, &mut rel) == 0).then_some(class);
        if std::env::var_os("NX_NOQOS").is_none() {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
        return had;
    }
    #[allow(unreachable_code)]
    None
}

/// Undo [`prefer_performance_cores`].
pub fn restore_qos(class: Option<u32>) {
    #[cfg(target_os = "macos")]
    if let Some(c) = class {
        // SAFETY: sets the calling thread's own QoS class.
        unsafe {
            pthread_set_qos_class_self_np(c, 0);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = class;
}

#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_self() -> *mut std::ffi::c_void;
    fn pthread_get_qos_class_np(thread: *mut std::ffi::c_void, class: *mut u32, relative: *mut i32) -> i32;
    fn pthread_set_qos_class_self_np(class: u32, relative: i32) -> i32;
}

/// Hand memory the allocator is caching back to the system. Loading
/// repacks every weight matrix through short-lived buffers, and the
/// allocator keeps those pages unless asked (the native engine does this
/// too).
pub fn release_cached_memory() {
    #[cfg(target_os = "macos")]
    // SAFETY: a null zone means every zone; the goal 0 means "as much as
    // possible". The call only returns free pages.
    unsafe {
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

/// Size the global rayon pool to the performance cores, unless the caller
/// already configured it (or set `RAYON_NUM_THREADS`).
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Without threads rayon runs on the calling thread by itself.
        if !cfg!(target_arch = "wasm32") && std::env::var_os("RAYON_NUM_THREADS").is_none() {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(performance_cores()).build_global();
        }
    });
}

/// `getrusage(RUSAGE_SELF)`: `timeval`s as `[sec, usec]` words.
#[repr(C)]
pub(crate) struct Rusage {
    pub utime: [i64; 2],
    pub stime: [i64; 2],
    pub maxrss: i64,
    rest: [i64; 13],
}

pub(crate) fn rusage() -> Option<Rusage> {
    #[cfg(unix)]
    {
        let mut ru = std::mem::MaybeUninit::<Rusage>::zeroed();
        // SAFETY: getrusage fills a plain struct we own.
        if unsafe { getrusage(0, ru.as_mut_ptr()) } == 0 {
            // SAFETY: filled above.
            return Some(unsafe { ru.assume_init() });
        }
    }
    None
}

#[cfg(unix)]
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}
