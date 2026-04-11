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
        s.serialize_field("num_variables", &self.num_variables)?;
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

        let nv = data.num_variables as usize;
        if data.opcodes.len() != data.arg_indices.len() {
            return Err(serde::de::Error::custom(format!(
                "opcodes.len() ({}) != arg_indices.len() ({})",
                data.opcodes.len(),
                data.arg_indices.len()
            )));
        }
        if data.opcodes.len() != nv {
            return Err(serde::de::Error::custom(format!(
                "opcodes.len() ({}) != num_variables ({})",
                data.opcodes.len(),
                nv
            )));
        }
        if data.values.len() != nv {
            return Err(serde::de::Error::custom(format!(
                "values.len() ({}) != num_variables ({})",
                data.values.len(),
                nv
            )));
        }
        if data.num_inputs > data.num_variables {
            return Err(serde::de::Error::custom(format!(
                "num_inputs ({}) > num_variables ({})",
                data.num_inputs, data.num_variables
            )));
        }
        for i in 0..data.num_inputs as usize {
            if data.opcodes[i] != OpCode::Input {
                return Err(serde::de::Error::custom(format!(
                    "opcodes[{}] should be Input but is {:?}",
                    i, data.opcodes[i]
                )));
            }
        }
        if data.output_index != u32::MAX && data.output_index >= data.num_variables {
            return Err(serde::de::Error::custom(format!(
                "output_index ({}) >= num_variables ({})",
                data.output_index, data.num_variables
            )));
        }
        for (i, &oi) in data.output_indices.iter().enumerate() {
            if oi >= data.num_variables {
                return Err(serde::de::Error::custom(format!(
                    "output_indices[{}] ({}) >= num_variables ({})",
                    i, oi, data.num_variables
                )));
            }
        }
        for (i, &[a, b]) in data.arg_indices.iter().enumerate() {
            if data.opcodes[i] != OpCode::Input
                && data.opcodes[i] != OpCode::Const
                && data.opcodes[i] != OpCode::Custom
            {
                if a >= data.num_variables {
                    return Err(serde::de::Error::custom(format!(
                        "arg_indices[{}][0] ({}) >= num_variables ({})",
                        i, a, data.num_variables
                    )));
                }
                // b may be UNUSED (0xFFFFFFFF) for unary ops, or a powi exponent
                if b != u32::MAX && data.opcodes[i] != OpCode::Powi && b >= data.num_variables {
                    return Err(serde::de::Error::custom(format!(
                        "arg_indices[{}][1] ({}) >= num_variables ({})",
                        i, b, data.num_variables
                    )));
                }
            }
        }

        Ok(BytecodeTape {
            opcodes: data.opcodes,
            arg_indices: data.arg_indices,
            values: data.values,
            num_inputs: data.num_inputs,
            num_variables: data.num_variables,
            output_index: data.output_index,
            output_indices: data.output_indices,
            custom_ops: Vec::new(),
            custom_second_args: data.custom_second_args,
        })
    }
}
