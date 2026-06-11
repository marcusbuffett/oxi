//! Offline ZCA whitening estimation for trunk-mean position embeddings.
//!
//! Transformer trunk embeddings are anisotropic: all positions share a narrow
//! cone, so raw cosine similarities compress into a small range dominated by a
//! few high-variance directions (side to move, material, phase). Whitening
//! `y = normalize((x - mean) * Σ^(-1/2))` centers the cone and equalizes
//! variance across directions, spreading the cosine range and shifting
//! discrimination to the informative directions.
//!
//! Statistics are estimated from L2-normalized trunk-mean embeddings of a
//! training-stream sample and saved as `whitening.json` next to `model.mpk`
//! (see `inference::WhiteningTransform` for the serving side).

use crate::inference::WhiteningTransform;

/// Shrinkage applied to the covariance diagonal before inversion, as a
/// fraction of the average eigenvalue. Keeps near-null directions from
/// exploding under Σ^(-1/2) when the sample is small.
const EIGENVALUE_FLOOR_FRACTION: f64 = 1e-3;

/// Estimate a ZCA whitening transform from a corpus of (L2-normalized)
/// embeddings: `W = U diag((λ + ε)^(-1/2)) Uᵀ` from the eigendecomposition of
/// the sample covariance.
pub fn compute_whitening(embeddings: &[Vec<f32>]) -> anyhow::Result<WhiteningTransform> {
    anyhow::ensure!(!embeddings.is_empty(), "no embeddings to whiten");
    let d = embeddings[0].len();
    let n = embeddings.len();
    anyhow::ensure!(
        n > d,
        "need more samples ({n}) than dimensions ({d}) for a stable covariance"
    );

    let mut mean = vec![0.0f64; d];
    for emb in embeddings {
        for (m, &x) in mean.iter_mut().zip(emb) {
            *m += x as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Covariance of centered embeddings (upper triangle, then mirrored).
    let mut cov = vec![0.0f64; d * d];
    let mut centered = vec![0.0f64; d];
    for emb in embeddings {
        for ((c, &x), m) in centered.iter_mut().zip(emb).zip(&mean) {
            *c = x as f64 - m;
        }
        for i in 0..d {
            let ci = centered[i];
            if ci == 0.0 {
                continue;
            }
            let row = &mut cov[i * d..(i + 1) * d];
            for (j, &cj) in centered.iter().enumerate().skip(i) {
                row[j] += ci * cj;
            }
        }
    }
    for i in 0..d {
        for j in i..d {
            let v = cov[i * d + j] / (n as f64 - 1.0);
            cov[i * d + j] = v;
            cov[j * d + i] = v;
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&cov, d);
    let mean_eigenvalue = eigenvalues.iter().sum::<f64>() / d as f64;
    let epsilon = (mean_eigenvalue * EIGENVALUE_FLOOR_FRACTION).max(1e-12);

    // W = U diag((λ + ε)^(-1/2)) Uᵀ, row-major.
    let mut transform = vec![0.0f32; d * d];
    let inv_sqrt: Vec<f64> = eigenvalues
        .iter()
        .map(|&l| 1.0 / (l.max(0.0) + epsilon).sqrt())
        .collect();
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0f64;
            for k in 0..d {
                // eigenvectors stored column-major per eigenvector: U[i][k] = eigenvectors[i * d + k]
                acc += eigenvectors[i * d + k] * inv_sqrt[k] * eigenvectors[j * d + k];
            }
            transform[i * d + j] = acc as f32;
        }
    }

    Ok(WhiteningTransform {
        dim: d,
        samples: n,
        mean: mean.iter().map(|&m| m as f32).collect(),
        transform,
    })
}

/// Cyclic Jacobi eigendecomposition for a symmetric matrix (row-major d×d).
/// Returns (eigenvalues, eigenvectors) with eigenvector k stored in column k
/// (i.e. `eigenvectors[i * d + k]` is component i of eigenvector k).
fn jacobi_eigen_symmetric(matrix: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }

    const MAX_SWEEPS: usize = 30;
    for _sweep in 0..MAX_SWEEPS {
        let mut off_diagonal = 0.0f64;
        for i in 0..d {
            for j in (i + 1)..d {
                off_diagonal += a[i * d + j] * a[i * d + j];
            }
        }
        if off_diagonal.sqrt() < 1e-11 {
            break;
        }

        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-14 {
                    continue;
                }
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..d).map(|i| a[i * d + i]).collect();
    (eigenvalues, v)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-9)
}

pub fn percentiles(values: &mut [f32]) -> [f32; 7] {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| values[((values.len() - 1) as f64 * q) as usize];
    [
        pick(0.0),
        pick(0.05),
        pick(0.25),
        pick(0.5),
        pick(0.75),
        pick(0.95),
        pick(1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_recovers_known_eigenvalues() {
        // Symmetric 3x3 with known spectrum: diag(1,2,3) rotated is overkill;
        // use a simple matrix with analytically known eigenvalues.
        // [[2,1,0],[1,2,0],[0,0,5]] has eigenvalues 1, 3, 5.
        let m = vec![2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 5.0];
        let (mut vals, _) = jacobi_eigen_symmetric(&m, 3);
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 1.0).abs() < 1e-9);
        assert!((vals[1] - 3.0).abs() < 1e-9);
        assert!((vals[2] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn whitening_decorrelates_samples() {
        // Full-rank anisotropic data (deterministic LCG noise); after
        // whitening, the sample covariance of transformed vectors should be
        // ~identity.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mut embeddings = Vec::new();
        for _ in 0..2000 {
            let (u, v, w) = (uniform(), uniform(), uniform());
            // Correlated dims with all eigenvalues well above the eigenvalue
            // floor (the floor deliberately under-whitens near-null
            // directions, so near-singular test data would not reach identity;
            // verified against numpy).
            embeddings.push(vec![u, 0.6 * u + 0.8 * v, 0.5 * w]);
        }
        let w = compute_whitening(&embeddings).unwrap();
        let transformed: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| {
                let d = w.dim;
                let centered: Vec<f32> = e.iter().zip(&w.mean).map(|(x, m)| x - m).collect();
                let mut out = vec![0.0f32; d];
                for (i, &c) in centered.iter().enumerate() {
                    for j in 0..d {
                        out[j] += c * w.transform[i * d + j];
                    }
                }
                out
            })
            .collect();
        let n = transformed.len() as f64;
        for i in 0..3 {
            for j in 0..3 {
                let cov: f64 = transformed
                    .iter()
                    .map(|t| t[i] as f64 * t[j] as f64)
                    .sum::<f64>()
                    / (n - 1.0);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cov - expected).abs() < 0.05,
                    "cov[{i}][{j}] = {cov}, expected {expected}"
                );
            }
        }
    }
}
