//! Trait implementations for AD types (`std::ops`, `num_traits`, `simba`).

mod dual_vec_num_traits;
mod dual_vec_ops;
mod num_traits_impls;
mod std_ops;

#[cfg(feature = "simba")]
mod simba_impls;

#[cfg(feature = "bytecode")]
mod breverse_num_traits;
#[cfg(feature = "bytecode")]
mod breverse_ops;

#[cfg(feature = "taylor")]
mod taylor_num_traits;
#[cfg(feature = "taylor")]
mod taylor_std_ops;

#[cfg(feature = "laurent")]
mod laurent_num_traits;
#[cfg(feature = "laurent")]
mod laurent_std_ops;
