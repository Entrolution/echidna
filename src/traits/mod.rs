//! Trait implementations for AD types (`std::ops`, `num_traits`, `simba`).

pub mod dual_vec_num_traits;
pub mod dual_vec_ops;
pub mod num_traits_impls;
pub mod std_ops;

#[cfg(feature = "simba")]
pub mod simba_impls;

#[cfg(feature = "bytecode")]
pub mod breverse_num_traits;
#[cfg(feature = "bytecode")]
pub mod breverse_ops;

#[cfg(feature = "taylor")]
pub mod taylor_num_traits;
#[cfg(feature = "taylor")]
pub mod taylor_std_ops;

#[cfg(feature = "laurent")]
pub mod laurent_num_traits;
#[cfg(feature = "laurent")]
pub mod laurent_std_ops;
