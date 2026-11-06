//! The boolean circuit agrees with Plonky3's native permutation, including on non-canonical inputs.
#![cfg(feature = "poseidon2-koalabear")]

use mpz_circuits_core::circuits::poseidon2_koalabear::permute;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_koala_bear::{KoalaBear, default_koalabear_poseidon2_16};
use p3_symmetric::Permutation;

#[test]
fn matches_plonky3() {
    let circuit = permute().unwrap();
    let native = default_koalabear_poseidon2_16();
    let mut seed = 0x1234_5678_9abc_def0u64;
    for case in 0..16 {
        let words: [u32; 16] = std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            match case {
                0 => 0,
                1 => u32::MAX,
                _ => seed as u32,
            }
        });
        let bits = words.iter().flat_map(|w| (0..32).map(move |i| w >> i & 1 == 1));
        let output = circuit.evaluate(bits).unwrap();
        let got = output
            .chunks(32)
            .map(|bits| bits.iter().rev().fold(0, |word, bit| word << 1 | u32::from(*bit)))
            .collect::<Vec<_>>();
        let mut state = words.map(KoalaBear::from_u32);
        native.permute_mut(&mut state);
        assert_eq!(got, state.map(|x| x.as_canonical_u32()), "case {case}");
    }
}
