mod normal;

use pyo3::prelude::*;

use crate::normal::bsm;

#[pyclass(subclass)]
struct Mrg32k3a {
    rng: mrg32k3a_rs::Mrg32k3a,
}

#[pymethods]
impl Mrg32k3a {
    #[new]
    #[pyo3(signature = (
        seed=[12345, 12345, 12345, 12345, 12345, 12345],
        s_ss_sss_index=None
    ))]
    fn new(seed: [u32; 6], s_ss_sss_index: Option<[usize; 3]>) -> Self {
        let stream_indices = s_ss_sss_index.unwrap_or([0, 0, 0]);
        let rng = mrg32k3a_rs::Mrg32k3a::new(&seed, &stream_indices);
        Mrg32k3a { rng }
    }

    fn random(&mut self) -> f64 {
        self.rng.next_f64()
    }

    #[getter]
    fn _current_state(&self) -> (u32, u32, u32, u32, u32, u32) {
        self.rng.get_state().into()
    }

    #[setter(_current_state)]
    fn set_current_state(&mut self, state: (u32, u32, u32, u32, u32, u32)) {
        self.rng.set_state(state.into());
    }

    fn set_state(&mut self, state: (u32, u32, u32, u32, u32, u32)) {
        self.rng.set_state(state.into());
    }

    fn get_current_state(&self) -> (u32, u32, u32, u32, u32, u32) {
        self.rng.get_state().into()
    }

    #[getter]
    fn s_ss_sss_index(&self) -> [usize; 3] {
        self.rng.get_stream_indices()
    }

    #[getter]
    fn stream_start(&self) -> (u32, u32, u32, u32, u32, u32) {
        self.rng.get_stream_start().into()
    }

    #[getter]
    fn substream_start(&self) -> (u32, u32, u32, u32, u32, u32) {
        self.rng.get_substream_start().into()
    }

    #[getter]
    fn subsubstream_start(&self) -> (u32, u32, u32, u32, u32, u32) {
        self.rng.get_subsubstream_start().into()
    }

    fn advance_stream(&mut self) {
        self.rng.next_stream()
    }

    fn advance_substream(&mut self) {
        self.rng.next_substream()
    }

    fn advance_subsubstream(&mut self) {
        self.rng.next_subsubstream()
    }

    fn reset_stream(&mut self) {
        self.rng.reset_stream()
    }

    fn reset_substream(&mut self) {
        self.rng.reset_substream()
    }

    fn reset_subsubstream(&mut self) {
        self.rng.reset_subsubstream()
    }

    fn start_fixed_s_ss_sss(&mut self, s_ss_sss_triplet: [usize; 3]) {
        self.rng.set_indices(&s_ss_sss_triplet)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn mrg32k3a_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mrg32k3a>()?;
    m.add_function(wrap_pyfunction!(bsm, m)?)?;
    Ok(())
}
