//! Bit-exact numpy `RandomState` (MT19937) for the fixed permutations the
//! architecture derives from seeds (`_hada_perms`).

pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    /// `np.random.RandomState(seed)` for an integer seed (`init_genrand`).
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Self { mt, idx: 624 }
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        for i in 0..624 {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
            let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.mt[i] = v;
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// numpy legacy `random_interval(max)`: masked rejection sampling.
    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `RandomState.permutation(n)`: Fisher-Yates from the top.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut arr: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u32) as usize;
            arr.swap(i, j);
        }
        arr
    }
}

/// `_hada_perms(n, split)`: the two fixed Hadamard MLP permutations.
pub fn hada_perms(n: usize, split: bool) -> [Vec<usize>; 2] {
    crate::config::HADA_PERM_SEEDS.map(|s| {
        if split {
            let h = n / 2;
            let mut p = Mt19937::new(s).permutation(h);
            p.extend(Mt19937::new(s + 977).permutation(h).into_iter().map(|x| h + x));
            p
        } else {
            Mt19937::new(s).permutation(n)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_matches_numpy() {
        // np.random.RandomState(11).permutation(16)
        assert_eq!(Mt19937::new(11).permutation(16), vec![8, 3, 6, 10, 15, 4, 5, 14, 2, 13, 12, 7, 1, 11, 0, 9]);
    }
}
