//! `tlsn/poseidon2/koalabear/16/v1` over authenticated byte references: `DOMAIN ‖ input ‖ 0x01`,
//! zero-filled to whole 24-byte blocks, as little-endian 3-byte words absorbed by the
//! overwrite-mode, rate-8 sponge of the stock width-16 KoalaBear Poseidon2. The marker byte makes
//! the padding injective, so no capacity tweak is needed.

use std::sync::{Arc, LazyLock};

use mpz_circuits::{BuilderError, Circuit, circuits::poseidon2_koalabear::permute};
use mpz_vm_core::{
    Call, CallError, CallableExt, Vm, VmError,
    memory::{
        Array, FromRaw, MemoryExt, Slice, ToRaw, Vector, ViewExt,
        binary::{Binary, U8, U32},
    },
};

/// Domain prefix: 30 bytes, so the input starts on a word boundary.
pub const DOMAIN: &[u8; 30] = b"tlsn/poseidon2/koalabear/16/v1";

static PERMUTE: LazyLock<Result<Arc<Circuit>, BuilderError>> =
    LazyLock::new(|| permute().map(Arc::new));

/// Errors at the authenticated byte-hash boundary.
#[derive(Debug, thiserror::Error)]
pub enum Poseidon2Error {
    /// VM operation failed.
    #[error("KoalaBear VM: {0}")]
    Vm(#[from] VmError),
    /// The permutation circuit failed to build.
    #[error("KoalaBear circuit: {0}")]
    Circuit(&'static BuilderError),
    /// Circuit arguments have an incorrect length.
    #[error("KoalaBear call: {0}")]
    Call(#[from] CallError),
}

/// Partition-independent byte hash; the caller owns the input references.
#[derive(Default)]
pub struct Poseidon2KoalaBear {
    inputs: Vec<Slice>,
}

fn bytes(mut input: Slice) -> impl Iterator<Item = Slice> {
    std::iter::from_fn(move || {
        if input.len() == 0 {
            return None;
        }
        // PROOF: input references and every remainder contain whole bytes.
        let (byte, rest) = input.split_at(8);
        input = rest;
        Some(byte)
    })
}

impl Poseidon2KoalaBear {
    /// Appends bytes without introducing a partition delimiter.
    pub fn update(&mut self, input: &Vector<U8>) {
        self.inputs.push(input.to_raw());
    }

    /// Absorbs the framed input block by block and returns the 8 rate words, little-endian.
    pub fn finalize(self, vm: &mut dyn Vm<Binary>) -> Result<Array<U8, 32>, Poseidon2Error> {
        let circuit = PERMUTE.as_ref().map_err(Poseidon2Error::Circuit)?;
        let prefix = vm.alloc_vec::<U8>(DOMAIN.len())?;
        vm.mark_public(prefix)?;
        vm.assign(prefix, DOMAIN.to_vec())?;
        vm.commit(prefix)?;
        let zeros: Array<U32, 16> = vm.alloc()?;
        vm.mark_public(zeros)?;
        vm.assign(zeros, [0u32; 16])?;
        vm.commit(zeros)?;
        let one: U32 = vm.alloc()?;
        vm.mark_public(one)?;
        vm.assign(one, 1u32)?;
        vm.commit(one)?;
        // PROOF: U32 storage is LSB0, so its first eight bits are its low byte.
        let zero = zeros.to_raw().split_at(8).0;
        let marker = one.to_raw().split_at(8).0;
        let mut framed = bytes(prefix.to_raw())
            .chain(self.inputs.into_iter().flat_map(bytes))
            .chain([marker])
            .collect::<Vec<_>>();
        framed.resize(framed.len().next_multiple_of(24), zero);
        let mut state = zeros;
        for block in framed.chunks(24) {
            let mut call = Call::builder(circuit.clone());
            for word in block.chunks(3) {
                for byte in word {
                    call = call.arg(*byte);
                }
                call = call.arg(zero);
            }
            // PROOF: the state is sixteen U32 words; the capacity is the last eight.
            state = vm.call(call.arg(state.to_raw().split_at(8 * 32).1).build()?)?;
        }
        // PROOF: the first eight U32 words occupy exactly 32 bytes in Binary's LSB0 encoding.
        Ok(<Array<U8, 32> as FromRaw<Binary>>::from_raw(
            state.to_raw().split_at(8 * 32).0,
        ))
    }
}
