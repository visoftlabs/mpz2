//! Stock Plonky3 width-16 KoalaBear Poseidon2 over Boolean gates.
//!
//! Numbers are little-endian bit vectors with a tracked upper bound, so additions stay lazy and
//! reduction modulo `p = 2^31 − 2^24 + 1` happens only where a canonical value is needed: it folds
//! with `2^31 ≡ 2^24 − 1`, then subtracts `2^j·p` conditionally. Constant multiplications add
//! shifted copies for the set bits of `c` or of `p − c`, whichever is sparser.

use std::array::from_fn;

use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_koala_bear::{
    GenericPoseidon2LinearLayersKoalaBear, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
    KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_16_INTERNAL, KoalaBear,
};
use p3_poseidon2::GenericPoseidon2LinearLayers;

use crate::{BuilderError, Circuit, CircuitBuilder, Feed, Node, ops};

const P: u128 = KoalaBear::ORDER_U32 as u128;

fn width(max: u128) -> usize {
    (128 - max.leading_zeros() as usize).max(1)
}

/// A value in `0..=max`, as `width(max)` little-endian bits.
#[derive(Clone)]
struct Num {
    bits: Vec<Node<Feed>>,
    max: u128,
}

impl Num {
    fn constant(builder: &CircuitBuilder, value: u128) -> Self {
        let bits = (0..width(value))
            .map(|i| match value >> i & 1 {
                0 => builder.get_const_zero(),
                _ => builder.get_const_one(),
            })
            .collect();
        Self { bits, max: value }
    }

    fn bit(&self, builder: &CircuitBuilder, i: usize) -> Node<Feed> {
        self.bits.get(i).copied().unwrap_or(builder.get_const_zero())
    }

    fn add(&self, builder: &mut CircuitBuilder, other: &Self) -> Self {
        let max = self.max + other.max;
        let mut carry = builder.get_const_zero();
        let bits = (0..width(max))
            .map(|i| {
                let (a, b) = (self.bit(builder, i), other.bit(builder, i));
                let (sum, next) = ops::full_adder(builder, a, b, carry);
                carry = next;
                sum
            })
            .collect();
        // PROOF: the sum is at most `max`, so the final carry is zero.
        Self { bits, max }
    }

    fn shl(&self, builder: &CircuitBuilder, shift: usize) -> Self {
        let zeros = std::iter::repeat_n(builder.get_const_zero(), shift);
        Self {
            bits: zeros.chain(self.bits.iter().copied()).collect(),
            max: self.max << shift,
        }
    }

    /// `M·p − self` for the least `M` with `M·p ≥ 2^w`: the complement `2^w − 1 − self` plus
    /// the constant `M·p − 2^w + 1`; congruent to `−self`.
    fn negate(&self, builder: &mut CircuitBuilder) -> Self {
        let w = width(self.max);
        let multiple = (1u128 << w).div_ceil(P) * P;
        let inverted = Self {
            bits: (0..w)
                .map(|i| {
                    let bit = self.bit(builder, i);
                    builder.add_inv_gate(bit)
                })
                .collect(),
            max: (1 << w) - 1,
        };
        let offset = Self::constant(builder, multiple - (1 << w) + 1);
        Self {
            max: multiple,
            ..inverted.add(builder, &offset)
        }
    }

    /// `low + high·2^24 − high`, congruent to `low + high·2^31`.
    fn fold(&self, builder: &mut CircuitBuilder) -> Self {
        let low = Self {
            bits: self.bits.iter().take(31).copied().collect(),
            max: self.max.min((1 << 31) - 1),
        };
        let high = Self {
            bits: self.bits.iter().skip(31).copied().collect(),
            max: self.max >> 31,
        };
        let shifted = high.shl(builder, 24);
        let negated = high.negate(builder);
        low.add(builder, &shifted).add(builder, &negated)
    }

    /// Subtracts `c` when the value is at least `c`; requires the value below `2c`.
    fn subtract_if_above(&self, builder: &mut CircuitBuilder, c: u128) -> Self {
        let w = width(self.max);
        let constant = Self::constant(builder, c);
        let a = (0..w).map(|i| self.bit(builder, i)).collect::<Vec<_>>();
        let b = (0..w).map(|i| constant.bit(builder, i)).collect::<Vec<_>>();
        let (difference, borrow) = ops::wrapping_sub(builder, &a, &b);
        let bits = a
            .iter()
            .zip(difference)
            .map(|(kept, reduced)| {
                let delta = builder.add_xor_gate(*kept, reduced);
                let selected = builder.add_and_gate(delta, borrow);
                builder.add_xor_gate(reduced, selected)
            })
            .take(width(c - 1))
            .collect();
        Self { bits, max: c - 1 }
    }

    /// The canonical representative, 31 bits.
    fn reduce(&self, builder: &mut CircuitBuilder) -> Self {
        let mut value = self.clone();
        // PROOF: a fold maps `max ≥ 2^33` to about `max/2^7 + 3·2^31 < max`; the fixpoint is below 2^33.
        while value.max >> 33 != 0 {
            value = value.fold(builder);
        }
        for j in (0..3).rev() {
            if value.max >= P << j {
                // PROOF: the value is below `2^(j+1)·p`, from `max < 2^33 < 8p` and the steps above.
                value = value.subtract_if_above(builder, P << j);
            }
        }
        value.bits.resize(31, builder.get_const_zero());
        value
    }

    /// Schoolbook product of two canonical values.
    fn mul(&self, builder: &mut CircuitBuilder, other: &Self) -> Self {
        let mut product = vec![builder.get_const_zero(); 62];
        for (shift, bit) in other.bits.iter().enumerate() {
            let mut carry = builder.get_const_zero();
            for (i, a) in self.bits.iter().enumerate() {
                let partial = builder.add_and_gate(*a, *bit);
                let slot = product
                    .get(shift + i)
                    .copied()
                    .unwrap_or(builder.get_const_zero());
                let (sum, next) = ops::full_adder(builder, slot, partial, carry);
                if let Some(slot) = product.get_mut(shift + i) {
                    *slot = sum;
                }
                carry = next;
            }
            if let Some(slot) = product.get_mut(shift + self.bits.len()) {
                *slot = carry;
            }
        }
        Self {
            bits: product,
            max: self.max * other.max,
        }
    }

    /// `c·self` up to a multiple of p, from the sparser of `c` and `p − c`.
    fn mul_const(&self, builder: &mut CircuitBuilder, c: u32) -> Self {
        let negative = KoalaBear::ORDER_U32 - c;
        let (factor, negate) = match c.count_ones() <= negative.count_ones() {
            true => (c, false),
            false => (negative, true),
        };
        let terms = (0..32)
            .filter(|i| factor >> i & 1 == 1)
            .map(|i| self.shl(builder, i))
            .collect::<Vec<_>>();
        let mut terms = terms.into_iter();
        let Some(first) = terms.next() else {
            return Self::constant(builder, 0);
        };
        let sum = terms.fold(first, |sum, term| sum.add(builder, &term));
        match negate {
            true => sum.negate(builder),
            false => sum,
        }
    }

    fn cube(&self, builder: &mut CircuitBuilder) -> Self {
        let square = self.mul(builder, self).reduce(builder);
        square.mul(builder, self).reduce(builder)
    }
}

/// Plonky3's `MDSMat4` per 4-word block, then the block sums; outputs reduced.
fn external(builder: &mut CircuitBuilder, state: &mut [Num; 16]) {
    for x in state.as_chunks_mut::<4>().0 {
        let t01 = x[0].add(builder, &x[1]);
        let t23 = x[2].add(builder, &x[3]);
        let sum = t01.add(builder, &t23);
        let s1 = sum.add(builder, &x[1]);
        let s3 = sum.add(builder, &x[3]);
        let d0 = x[0].shl(builder, 1);
        let d2 = x[2].shl(builder, 1);
        *x = [
            s1.add(builder, &t01),
            s1.add(builder, &d2),
            s3.add(builder, &t23),
            s3.add(builder, &d0),
        ];
    }
    let sums: [Num; 4] = from_fn(|i| {
        let a = state[i].add(builder, &state[i + 4]);
        let b = state[i + 8].add(builder, &state[i + 12]);
        a.add(builder, &b)
    });
    for (i, word) in state.iter_mut().enumerate() {
        *word = word.add(builder, &sums[i % 4]).reduce(builder);
    }
}

fn full_rounds(builder: &mut CircuitBuilder, state: &mut [Num; 16], rounds: &[[KoalaBear; 16]]) {
    for constants in rounds {
        for (word, constant) in state.iter_mut().zip(constants) {
            let constant = Num::constant(builder, constant.as_canonical_u32().into());
            *word = word.add(builder, &constant).reduce(builder).cube(builder);
        }
        external(builder, state);
    }
}

/// `fn([u32; 16]) -> [u32; 16]`; inputs are reduced modulo p, outputs are canonical.
pub fn permute() -> Result<Circuit, BuilderError> {
    let mut builder = CircuitBuilder::new();
    let input: [[_; 32]; 16] = from_fn(|_| from_fn(|_| builder.add_input()));
    let mut state = input.map(|bits| {
        Num {
            bits: bits.to_vec(),
            max: u32::MAX.into(),
        }
        .reduce(&mut builder)
    });
    external(&mut builder, &mut state);
    full_rounds(
        &mut builder,
        &mut state,
        &KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
    );

    let mut diagonal = [KoalaBear::ONE; 16];
    GenericPoseidon2LinearLayersKoalaBear::internal_linear_layer(&mut diagonal);
    let diagonal = diagonal.map(|value| (value - KoalaBear::from_u32(16)).as_canonical_u32());
    for constant in KOALABEAR_POSEIDON2_RC_16_INTERNAL {
        let constant = Num::constant(&builder, constant.as_canonical_u32().into());
        state[0] = state[0]
            .add(&mut builder, &constant)
            .reduce(&mut builder)
            .cube(&mut builder);
        let zero = Num::constant(&builder, 0);
        let sum = state
            .iter()
            .fold(zero, |sum, word| sum.add(&mut builder, word));
        for (word, coefficient) in state.iter_mut().zip(diagonal) {
            *word = word
                .mul_const(&mut builder, coefficient)
                .add(&mut builder, &sum)
                .reduce(&mut builder);
        }
    }
    full_rounds(
        &mut builder,
        &mut state,
        &KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
    );
    for word in state {
        for bit in word.bits {
            let output = builder.add_id_gate(bit);
            builder.add_output(output);
        }
        // PROOF: the builder requires an actual gate for each output, including zero.
        let zero = builder.add_xor_gate(input[0][0], input[0][0]);
        builder.add_output(zero);
    }
    builder.build()
}
