use std::collections::HashMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::float::Float;
use crate::opcode::OpCode;

use super::BytecodeTape;

impl<F: Float + Serialize> Serialize for BytecodeTape<F> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !self.custom_ops.is_empty() {
            return Err(serde::ser::Error::custom(
                "cannot serialize a BytecodeTape containing custom ops; \
                 custom ops must be re-registered after deserialization",
            ));
        }
        let mut s = serializer.serialize_struct("BytecodeTape", 8)?;
        s.serialize_field("opcodes", &self.opcodes)?;
        s.serialize_field("arg_indices", &self.arg_indices)?;
        s.serialize_field("values", &self.values)?;
        s.serialize_field("num_inputs", &self.num_inputs)?;
        // The tape no longer stores an entry count, but the wire format keeps
        // the field so existing payloads round-trip unchanged.
        s.serialize_field("num_variables", &(self.opcodes.len() as u32))?;
        s.serialize_field("output_index", &self.output_index)?;
        s.serialize_field("output_indices", &self.output_indices)?;
        s.serialize_field("custom_second_args", &self.custom_second_args)?;
        s.end()
    }
}

impl<'de, F: Float + Deserialize<'de>> Deserialize<'de> for BytecodeTape<F> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct TapeData<F> {
            opcodes: Vec<OpCode>,
            arg_indices: Vec<[u32; 2]>,
            values: Vec<F>,
            num_inputs: u32,
            num_variables: u32,
            output_index: u32,
            #[serde(default)]
            output_indices: Vec<u32>,
            #[serde(default)]
            custom_second_args: HashMap<u32, u32>,
        }

        let data = TapeData::<F>::deserialize(deserializer)?;

        // Custom ops carry Rust callbacks that cannot be serialized, so a
        // payload containing them (or their operand side table) can never
        // be restored to a working tape. These scans touch only the field
        // they inspect — nothing here may index one array by another's
        // length before `validate()` has confirmed the lengths agree.
        if let Some(i) = data.opcodes.iter().position(|&op| op == OpCode::Custom) {
            return Err(serde::de::Error::custom(format!(
                "opcodes[{i}] is Custom, which cannot be deserialized (custom ops have no serializable callback)"
            )));
        }
        if !data.custom_second_args.is_empty() {
            return Err(serde::de::Error::custom(
                "custom_second_args must be empty: custom ops cannot be deserialized",
            ));
        }

        // The in-memory tape derives its entry count from `opcodes.len()`,
        // so the payload's `num_variables` exists only as a consistency
        // stamp — reject a mismatch here, since `validate()` can no longer
        // see the transmitted value.
        if data.num_variables as usize != data.opcodes.len() {
            return Err(serde::de::Error::custom(format!(
                "num_variables ({}) != opcodes.len() ({})",
                data.num_variables,
                data.opcodes.len()
            )));
        }

        let tape = BytecodeTape {
            opcodes: data.opcodes,
            arg_indices: data.arg_indices,
            values: data.values,
            num_inputs: data.num_inputs,
            output_index: data.output_index,
            output_indices: data.output_indices,
            custom_ops: Vec::new(),
            custom_second_args: data.custom_second_args,
            #[cfg(debug_assertions)]
            tape_id: crate::bytecode_tape::next_tape_id(),
        };
        // Full structural validation: buffer lengths, the Input prefix,
        // output indices, per-op operand bounds (strictly-earlier slots),
        // and side-table consistency. Rejecting here turns every corrupt
        // payload into a clean error instead of a panic (or silently wrong
        // derivatives) at evaluation time.
        tape.validate().map_err(serde::de::Error::custom)?;
        Ok(tape)
    }
}
