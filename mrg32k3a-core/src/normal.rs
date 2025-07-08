use pyo3::prelude::*;

const BSM_A: [f64; 4] = [
    2.50662823884,
    -18.61500062529,
    41.39119773534,
    -25.44106049637,
];

const BSM_B: [f64; 5] = [
    1.0,
    -8.47351093090,
    23.08336743743,
    -21.06224101826,
    3.13082909833,
];

const BSM_C: [f64; 9] = [
    0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187,
];

fn horner(x: f64, coefficients: &[f64]) -> f64 {
    let mut result = 0.0;
    for &c in coefficients.iter().rev() {
        result = result * x + c;
    }
    result
}

#[pyfunction]
pub fn bsm(u: f64) -> f64 {
    if !(0.0 < u && u < 1.0) {
        panic!("input must be in (0, 1)");
    }

    // Center the input around 0 for symmetry
    let v = u - 0.5;

    // Use rational approximation for central region (most common case)
    if v.abs() < 0.42 {
        let v_squared = v * v;
        return v * horner(v_squared, &BSM_A) / horner(v_squared, &BSM_B);
    }

    // Use tail approximation for extreme values
    let (p, sign) = if v < 0.0 { (u, -1.0) } else { (1.0 - u, 1.0) };

    sign * horner((-p.ln()).ln(), &BSM_C)
}
