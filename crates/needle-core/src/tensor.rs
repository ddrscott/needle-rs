//! A minimal dense f32 tensor and the flat parameter map the checkpoints use.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

/// Row-major f32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, data }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(shape.to_vec(), vec![0.0; shape.iter().product()])
    }

    pub fn scalar(v: f32) -> Self {
        Self::new(vec![], vec![v])
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of one entry along axis 0.
    pub fn stride0(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    /// The `i`-th slice along axis 0.
    pub fn index0(&self, i: usize) -> Tensor {
        let s = self.stride0();
        Tensor::new(self.shape[1..].to_vec(), self.data[i * s..(i + 1) * s].to_vec())
    }

    pub fn slice0(&self, i: usize) -> &[f32] {
        let s = self.stride0();
        &self.data[i * s..(i + 1) * s]
    }

    /// Gather rows along axis 0.
    pub fn take0(&self, idx: &[usize]) -> Tensor {
        let s = self.stride0();
        let mut data = Vec::with_capacity(idx.len() * s);
        for &i in idx {
            data.extend_from_slice(&self.data[i * s..(i + 1) * s]);
        }
        let mut shape = self.shape.clone();
        shape[0] = idx.len();
        Tensor::new(shape, data)
    }

    pub fn reshape(mut self, shape: &[usize]) -> Result<Tensor> {
        if shape.iter().product::<usize>() != self.data.len() {
            bail!("cannot reshape {:?} to {:?}", self.shape, shape);
        }
        self.shape = shape.to_vec();
        Ok(self)
    }

    /// Transpose the last two axes.
    pub fn t_last2(&self) -> Tensor {
        let n = self.ndim();
        assert!(n >= 2);
        let (r, c) = (self.shape[n - 2], self.shape[n - 1]);
        let lead: usize = self.shape[..n - 2].iter().product();
        let mut out = vec![0.0; self.data.len()];
        for b in 0..lead {
            let src = &self.data[b * r * c..(b + 1) * r * c];
            let dst = &mut out[b * r * c..(b + 1) * r * c];
            for i in 0..r {
                for j in 0..c {
                    dst[j * r + i] = src[i * c + j];
                }
            }
        }
        let mut shape = self.shape.clone();
        shape.swap(n - 2, n - 1);
        Tensor::new(shape, out)
    }
}

/// Flat parameter map keyed by `/`-joined Flax paths (`stack/layers/block/...`).
pub type Params = BTreeMap<String, Tensor>;

pub fn get<'a>(params: &'a Params, name: &str) -> Result<&'a Tensor> {
    params.get(name).ok_or_else(|| anyhow::anyhow!("checkpoint is missing parameter {name}"))
}
