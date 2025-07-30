const M1: i64 = 4294967087;
const M1_PLUS_1: f64 = 4294967088.0;
const M2: i64 = 4294944443;
const A12: u32 = 1403580;
const A13: u32 = 810728;
const A21: u32 = 527612;
const A23: u32 = 1370589;

const A1P47: [[u32; 3]; 3] = [
    [1362557480, 3230022138, 4278720212],
    [3427386258, 3848976950, 3230022138],
    [2109817045, 2441486578, 3848976950],
];
const A2P47: [[u32; 3]; 3] = [
    [2920112852, 1965329198, 1177141043],
    [2135250851, 2920112852, 969184056],
    [296035385, 2135250851, 4267827987],
];

const A1P94: [[u32; 3]; 3] = [
    [2873769531, 2081104178, 596284397],
    [4153800443, 1261269623, 2081104178],
    [3967600061, 1830023157, 1261269623],
];
const A2P94: [[u32; 3]; 3] = [
    [1347291439, 2050427676, 736113023],
    [4102191254, 1347291439, 878627148],
    [1293500383, 4102191254, 745646810],
];

const A1P141: [[u32; 3]; 3] = [
    [3230096243, 2131723358, 3262178024],
    [2882890127, 4088518247, 2131723358],
    [3991553306, 1282224087, 4088518247],
];
const A2P141: [[u32; 3]; 3] = [
    [2196438580, 805386227, 4266375092],
    [4124675351, 2196438580, 2527961345],
    [94452540, 4124675351, 2825656399],
];

#[allow(clippy::needless_range_loop)]
fn matmul_mod(a: &[[u32; 3]; 3], b: &[[u32; 3]; 3], m: i64) -> [[u32; 3]; 3] {
    let mut result = [[0u32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0u128;
            for k in 0..3 {
                sum += a[i][k] as u128 * b[k][j] as u128;
            }
            result[i][j] = (sum % m as u128) as u32;
        }
    }
    result
}

fn power_mod(a: &[[u32; 3]; 3], j: usize, m: i64) -> [[u32; 3]; 3] {
    let mut b = [[0u32; 3]; 3];
    b[0][0] = 1;
    b[1][1] = 1;
    b[2][2] = 1;

    let mut a = *a;
    let mut j = j;

    while j > 0 {
        if j & 1 == 1 {
            b = matmul_mod(&a, &b, m);
        }
        a = matmul_mod(&a, &a, m);
        j >>= 1;
    }
    b
}

fn jump_state(a: &[[u32; 3]; 3], b: &[[u32; 3]; 3], state: &[u32; 6]) -> [u32; 6] {
    let state_a = [state[0], state[1], state[2]];
    let state_b = [state[3], state[4], state[5]];

    let new_state_a = [
        ((a[0][0] as u128 * state_a[0] as u128
            + a[0][1] as u128 * state_a[1] as u128
            + a[0][2] as u128 * state_a[2] as u128)
            % (M1 as u128)) as u32,
        ((a[1][0] as u128 * state_a[0] as u128
            + a[1][1] as u128 * state_a[1] as u128
            + a[1][2] as u128 * state_a[2] as u128)
            .rem_euclid(M1 as u128)) as u32,
        ((a[2][0] as u128 * state_a[0] as u128
            + a[2][1] as u128 * state_a[1] as u128
            + a[2][2] as u128 * state_a[2] as u128)
            .rem_euclid(M1 as u128)) as u32,
    ];

    let new_state_b = [
        ((b[0][0] as u128 * state_b[0] as u128
            + b[0][1] as u128 * state_b[1] as u128
            + b[0][2] as u128 * state_b[2] as u128)
            .rem_euclid(M2 as u128)) as u32,
        ((b[1][0] as u128 * state_b[0] as u128
            + b[1][1] as u128 * state_b[1] as u128
            + b[1][2] as u128 * state_b[2] as u128)
            .rem_euclid(M2 as u128)) as u32,
        ((b[2][0] as u128 * state_b[0] as u128
            + b[2][1] as u128 * state_b[1] as u128
            + b[2][2] as u128 * state_b[2] as u128)
            .rem_euclid(M2 as u128)) as u32,
    ];

    [
        new_state_a[0],
        new_state_a[1],
        new_state_a[2],
        new_state_b[0],
        new_state_b[1],
        new_state_b[2],
    ]
}

fn compute_stream_start(
    a1: &[[u32; 3]; 3],
    a2: &[[u32; 3]; 3],
    index: usize,
    state: &[u32; 6],
) -> [u32; 6] {
    let a1 = power_mod(a1, index, M1);
    let a2 = power_mod(a2, index, M2);
    jump_state(&a1, &a2, state)
}

#[derive(Clone)]
pub struct Mrg32k3a {
    state: [u32; 6],
    seed: [u32; 6],
    stream_start: [u32; 6],
    substream_start: [u32; 6],
    subsubstream_start: [u32; 6],
    stream_indices: [usize; 3],
}

impl Mrg32k3a {
    pub fn new(seed: &[u32; 6], stream_indices: &[usize; 3]) -> Self {
        let mut value = Self {
            state: *seed,
            seed: *seed,
            stream_start: *seed,
            substream_start: *seed,
            subsubstream_start: *seed,
            stream_indices: [0, 0, 0],
        };
        value.set_indices(stream_indices);
        value
    }

    pub fn get_state(&self) -> [u32; 6] {
        self.state
    }

    pub fn set_state(&mut self, state: [u32; 6]) {
        self.state = state;
    }

    pub fn get_stream_indices(&self) -> [usize; 3] {
        self.stream_indices
    }

    pub fn get_stream_start(&self) -> [u32; 6] {
        self.stream_start
    }

    pub fn get_substream_start(&self) -> [u32; 6] {
        self.substream_start
    }

    pub fn get_subsubstream_start(&self) -> [u32; 6] {
        self.subsubstream_start
    }

    pub fn next_f64(&mut self) -> f64 {
        let s0 = self.state[0];
        let s1 = self.state[1];
        let s2 = self.state[2];
        let s3 = self.state[3];
        let s4 = self.state[4];
        let s5 = self.state[5];

        let new_s2 = (A12 as i64 * s1 as i64 - A13 as i64 * s0 as i64).rem_euclid(M1);
        let new_s5 = (A21 as i64 * s5 as i64 - A23 as i64 * s3 as i64).rem_euclid(M2);

        self.state = [s1, s2, new_s2 as u32, s4, s5, new_s5 as u32];

        let mut diff = new_s2 - new_s5;
        diff -= M1 * ((diff - 1) >> 63);
        (diff as f64) / M1_PLUS_1
    }

    pub fn next_stream(&mut self) {
        let new_state = jump_state(&A1P141, &A2P141, &self.stream_start);
        self.state = new_state;
        self.stream_start = new_state;
        self.substream_start = new_state;
        self.subsubstream_start = new_state;
        self.stream_indices[0] += 1;
        self.stream_indices[1] = 0;
        self.stream_indices[2] = 0;
    }

    pub fn next_substream(&mut self) {
        let new_state = jump_state(&A1P94, &A2P94, &self.substream_start);
        self.state = new_state;
        self.substream_start = new_state;
        self.subsubstream_start = new_state;
        self.stream_indices[1] += 1;
        self.stream_indices[2] = 0;
    }

    pub fn next_subsubstream(&mut self) {
        let new_state = jump_state(&A1P47, &A2P47, &self.subsubstream_start);
        self.state = new_state;
        self.subsubstream_start = new_state;
        self.stream_indices[2] += 1;
    }

    pub fn reset_stream(&mut self) {
        let new_state = self.stream_start;
        self.state = new_state;
        self.substream_start = new_state;
        self.subsubstream_start = new_state;
        self.stream_indices[1] = 0;
        self.stream_indices[2] = 0;
    }

    pub fn reset_substream(&mut self) {
        let new_state = self.substream_start;
        self.state = new_state;
        self.subsubstream_start = new_state;
        self.stream_indices[2] = 0;
    }

    pub fn reset_subsubstream(&mut self) {
        self.state = self.subsubstream_start;
    }

    pub fn set_indices(&mut self, stream_indices: &[usize; 3]) {
        self.stream_indices = *stream_indices;
        let [stream, substream, subsubstream] = *stream_indices;
        self.stream_start = compute_stream_start(&A1P141, &A2P141, stream, &self.seed);
        self.substream_start = compute_stream_start(&A1P94, &A2P94, substream, &self.stream_start);
        self.subsubstream_start =
            compute_stream_start(&A1P47, &A2P47, subsubstream, &self.substream_start);
        self.state = self.subsubstream_start;
    }
}
