//! Bit-exact ports of the two random streams `needle finetune` consumes:
//! numpy `default_rng(seed)` (SeedSequence + PCG64) for the holdout split and
//! epoch shuffles, and JAX's threefry `PRNGKey` / `split` / `normal` for the
//! LoRA `A` init. Matching them lets a Rust run follow a Python run step by
//! step.

// ---------------------------------------------------------------- numpy

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL: usize = 4;

fn hashmix(value: u32, hc: &mut u32) -> u32 {
    let mut v = value ^ *hc;
    *hc = hc.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hc);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `np.random.SeedSequence(seed).generate_state(n, np.uint64)`.
pub fn seed_sequence_u64(seed: u64, n: usize) -> Vec<u64> {
    let mut entropy = vec![seed as u32];
    if seed >> 32 != 0 {
        entropy.push((seed >> 32) as u32);
    }
    let mut pool = [0u32; POOL];
    let mut hc = INIT_A;
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
    }
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hc));
            }
        }
    }
    for &e in entropy.iter().skip(POOL) {
        for p in pool.iter_mut() {
            *p = mix(*p, hashmix(e, &mut hc));
        }
    }
    let mut hb = INIT_B;
    let words: Vec<u32> = (0..n * 2)
        .map(|i| {
            let mut v = pool[i % POOL] ^ hb;
            hb = hb.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hb);
            v ^ (v >> XSHIFT)
        })
        .collect();
    (0..n).map(|i| words[2 * i] as u64 | ((words[2 * i + 1] as u64) << 32)).collect()
}

const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

/// numpy `Generator(PCG64(seed))`.
pub struct NpRng {
    state: u128,
    inc: u128,
    spare: Option<u32>,
}

impl NpRng {
    pub fn new(seed: u64) -> Self {
        let s = seed_sequence_u64(seed, 4);
        let initstate = ((s[0] as u128) << 64) | s[1] as u128;
        let initseq = ((s[2] as u128) << 64) | s[3] as u128;
        let mut r = Self { state: 0, inc: (initseq << 1) | 1, spare: None };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let n = self.next_u64();
        self.spare = Some((n >> 32) as u32);
        n as u32
    }

    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        loop {
            let v = if max <= 0xffff_ffff { (self.next_u32() as u64) & mask } else { self.next_u64() & mask };
            if v <= max {
                return v;
            }
        }
    }

    /// `rng.permutation(n)`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u64) as usize;
            a.swap(i, j);
        }
        a
    }
}

// ---------------------------------------------------------------- JAX

fn rotl(x: u32, r: u32) -> u32 {
    x.rotate_left(r)
}

/// Threefry-2x32 with 20 rounds (`threefry2x32_p`).
pub fn threefry2x32(key: [u32; 2], x: [u32; 2]) -> [u32; 2] {
    const R: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    let ks = [key[0], key[1], key[0] ^ key[1] ^ 0x1BD1_1BDA];
    let mut x0 = x[0].wrapping_add(ks[0]);
    let mut x1 = x[1].wrapping_add(ks[1]);
    for round in 0..5 {
        for &r in &R[round % 2] {
            x0 = x0.wrapping_add(x1);
            x1 = rotl(x1, r) ^ x0;
        }
        x0 = x0.wrapping_add(ks[(round + 1) % 3]);
        x1 = x1.wrapping_add(ks[(round + 2) % 3]).wrapping_add(round as u32 + 1);
    }
    [x0, x1]
}

/// `jax.random.PRNGKey(seed)`.
pub fn prng_key(seed: u64) -> [u32; 2] {
    [(seed >> 32) as u32, seed as u32]
}

/// `jax.random.split(key)` (partitionable threefry): `(key, sub)`.
pub fn split(key: [u32; 2]) -> ([u32; 2], [u32; 2]) {
    (threefry2x32(key, [0, 0]), threefry2x32(key, [0, 1]))
}

/// XLA's single-precision `ErfInv` (Giles' approximation).
fn erfinv_f32(x: f32) -> f32 {
    let w = -((1.0 - x) * (1.0 + x)).ln();
    let p = if w < 5.0 {
        let w = w - 2.5;
        let mut p = 2.810_226_4e-8_f32;
        for c in [
            3.432_739_4e-7,
            -3.523_387_7e-06,
            -4.391_506_5e-6,
            0.000_218_580_87,
            -0.001_253_725,
            -0.004_177_681_6,
            0.246_640_73,
            1.501_409_4,
        ] {
            p = c + p * w;
        }
        p
    } else {
        let w = w.sqrt() - 3.0;
        let mut p = -0.000_200_214_26_f32;
        for c in [
            0.000_100_950_56,
            0.001_349_343_2,
            -0.003_673_428_4,
            0.005_739_507_7,
            -0.007_622_461_3,
            0.009_438_870_5,
            1.001_674,
            2.832_976_8,
        ] {
            p = c + p * w;
        }
        p
    };
    if x.abs() == 1.0 { x * f32::INFINITY } else { p * x }
}

/// `jax.random.normal(key, shape, float32)` flattened.
pub fn normal(key: [u32; 2], n: usize) -> Vec<f32> {
    let lo = f32::from_bits((-1.0f32).to_bits() - 1); // nextafter(-1, 0)
    let hi = 1.0f32;
    (0..n)
        .map(|i| {
            let [b1, b2] = threefry2x32(key, [(i as u64 >> 32) as u32, i as u32]);
            let bits = b1 ^ b2;
            let f = f32::from_bits((bits >> 9) | 0x3f80_0000) - 1.0;
            let u = lo.max(f * (hi - lo) + lo);
            std::f32::consts::SQRT_2 * erfinv_f32(u)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefry_known_answer() {
        // Random123 known-answer vector for threefry2x32_20.
        assert_eq!(threefry2x32([0x13198a2e, 0x03707344], [0x243f6a88, 0x85a308d3]), [0xc4923a9c, 0x483df7a0]);
    }

    #[test]
    fn numpy_streams_match() {
        assert_eq!(seed_sequence_u64(0, 4), vec![0xdb2cd7e7b0f478be, 0xabf4641a2c71ba49, 0x20c6ed6d9d7b8d41, 0x2c4099de223c39d4]);
        assert_eq!(NpRng::new(0).permutation(12), vec![9, 2, 7, 4, 5, 11, 0, 3, 6, 10, 8, 1]);
        assert_eq!(NpRng::new(5).permutation(7), vec![1, 4, 2, 3, 6, 5, 0]);
    }

    #[test]
    fn jax_streams_match() {
        let (k, s) = split(prng_key(0));
        assert_eq!(k, [0x6b200159, 0x99ba4efe]);
        assert_eq!(s, [0x375f238f, 0xcddb151d]);
        let want = [-2.442_455_8_f32, -2.035_680_5, 0.205_544_23, -0.353_550_2, -0.761_974_04, -1.178_551_8];
        for (a, b) in normal(s, 6).iter().zip(want) {
            assert!((a - b).abs() <= 2e-6 * b.abs().max(1.0), "{a} vs {b}");
        }
    }
}
