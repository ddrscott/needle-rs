//! `optax.chain(clip_by_global_norm(1.0), adamw(warmup_cosine_decay_schedule(...)))`.

use crate::graph::LoraParams;

/// `optax.warmup_cosine_decay_schedule(0, peak, warmup, decay_steps, end=0)`.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub peak: f32,
    pub warmup: usize,
    pub decay_steps: usize,
}

impl Schedule {
    pub fn lr(&self, count: usize) -> f32 {
        if count < self.warmup {
            // linear_schedule(0, peak, warmup)
            return self.peak * (count.min(self.warmup) as f32 / self.warmup as f32);
        }
        let steps = self.decay_steps - self.warmup;
        let c = (count - self.warmup).min(steps);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * c as f32 / steps as f32).cos());
        self.peak * cosine
    }
}

pub struct AdamW {
    pub schedule: Schedule,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_norm: f32,
    mu: Vec<Vec<f32>>,
    nu: Vec<Vec<f32>>,
    count: usize,
}

impl AdamW {
    /// optax defaults: b1 0.9, b2 0.999, eps 1e-8, weight decay 1e-4.
    pub fn new(schedule: Schedule, params: &LoraParams) -> Self {
        let zeros: Vec<Vec<f32>> = params.tensors().map(|t| vec![0.0; t.len()]).collect();
        Self { schedule, b1: 0.9, b2: 0.999, eps: 1e-8, weight_decay: 1e-4, max_norm: 1.0, mu: zeros.clone(), nu: zeros, count: 0 }
    }

    /// Apply one update in place; returns the pre-clip gradient norm.
    pub fn update(&mut self, params: &mut LoraParams, grads: &LoraParams) -> f32 {
        let norm = grads.tensors().flat_map(|g| g.iter()).map(|v| v * v).sum::<f32>().sqrt();
        let clip = if norm < self.max_norm { None } else { Some(self.max_norm / norm) };
        let lr = self.schedule.lr(self.count);
        self.count += 1;
        let (b1, b2) = (self.b1, self.b2);
        let bc1 = 1.0 - b1.powi(self.count as i32);
        let bc2 = 1.0 - b2.powi(self.count as i32);
        for (((p, g), mu), nu) in params.tensors_mut().zip(grads.tensors()).zip(self.mu.iter_mut()).zip(self.nu.iter_mut()) {
            for i in 0..p.len() {
                let gi = match clip {
                    Some(_) => (g[i] / norm) * self.max_norm,
                    None => g[i],
                };
                mu[i] = b1 * mu[i] + (1.0 - b1) * gi;
                nu[i] = b2 * nu[i] + (1.0 - b2) * gi * gi;
                let mh = mu[i] / bc1;
                let nh = nu[i] / bc2;
                let u = mh / (nh.sqrt() + self.eps) + self.weight_decay * p[i];
                p[i] -= lr * u;
            }
        }
        norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_shape() {
        let s = Schedule { peak: 1e-4, warmup: 1, decay_steps: 20 };
        assert_eq!(s.lr(0), 0.0);
        assert!((s.lr(1) - 1e-4).abs() < 1e-12);
        assert!(s.lr(10) < 1e-4 && s.lr(10) > 0.0);
        assert!(s.lr(20).abs() < 1e-12);
    }
}
