//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_PARALLEL_REDUCE_HPP
#define KOKKOS_PARALLEL_REDUCE_HPP

#include <Kokkos_ReductionIdentity.hpp>
#include <Kokkos_View.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Tools_Generic.hpp>
#include <type_traits>

namespace Kokkos {

namespace Impl {
//! Type of the reduction target.
enum class ReductionTargetType {
  SCALAR,
  VIEW
};

//! This class wraps the reduction target.
template <class ViewType, ReductionTargetType TargetType>
struct ReductionTarget;

//! Specialization for a scalar.
template <class ViewType>
struct ReductionTarget<ViewType, ReductionTargetType::SCALAR>
{
  using value_type = typename ViewType::value_type;
  using view_type = ViewType;

  static constexpr ReductionTargetType type = ReductionTargetType::SCALAR;

  KOKKOS_FUNCTION
  ReductionTarget(typename ViewType::value_type& value) : data(&value) {}

  ViewType data;
};

//! Specialization for a view.
template <class ViewType>
struct ReductionTarget<ViewType, ReductionTargetType::VIEW>
{
  using value_type = typename ViewType::value_type;
  using view_type = ViewType;

  static constexpr ReductionTargetType type = ReductionTargetType::VIEW;

  /**
   * Support for backward compatibility. But to me, this should be deprecated because it's a bad semantic.
   * Sometimes (even in Kokkos) people do
   *    int value;
   *    Kokkos::Sum<int, Kokkos::Cuda>(value);
   * So it seems such a code wants to target a scalar, but the new semantic we want ends up using the
   * reduction target type 'VIEW'. It happens usually when the scalar is on device, so we don't want the result
   * view type to be on host space.
   * Such a code should probably be changed to:
   *    int value;
   *    Kokkos::Sum<int, Kokkos::Cuda>(&value);
   * so we clearly see that we pass a pointer (and create an unmanaged view around 'value').
   */
  KOKKOS_FUNCTION
  ReductionTarget(value_type& value) : data(&value) {}

  template <typename T>
  KOKKOS_FUNCTION
  ReductionTarget(T&& value) : data(std::forward<T>(value)) {}

  ViewType data;
};

//! Concept for "is a reduction target type".
template <typename>
struct is_reduction_target : std::false_type {};

template <typename ViewType, ReductionTargetType TargetType>
struct is_reduction_target<ReductionTarget<ViewType, TargetType>> : std::true_type {};

template <typename T>
constexpr bool is_reduction_target_v = is_reduction_target<T>::value;

} // namespace Impl

/// Helper to define the deduction guides for the @c reducer.
/// Guides must work for:
///   - a scalar, in which case we reduce on a view whose memory space is @c Kokkos::HostSpace
///     @note It is probably useless.
///   - a view
#define DEDUCTION_GUIDE_FOR_REDUCER(reducer)                                                                         \
  template <typename Scalar, typename... Properties>                                                                 \
  KOKKOS_DEDUCTION_GUIDE                                                                                             \
  reducer(Kokkos::View<Scalar, Properties...> const&)                                                                \
      -> reducer<Scalar, typename Kokkos::View<Scalar, Properties...>::memory_space>;                                \
  template <typename Scalar, typename = std::enable_if_t<! Kokkos::is_view_v<Kokkos::Impl::remove_cvref_t<Scalar>>>> \
  KOKKOS_DEDUCTION_GUIDE reducer(Scalar&)                                                                            \
      -> reducer<Scalar>;

//! Same, for reducers that define a custom type.
#define DEDUCTION_GUIDE_FOR_REDUCER_CUSTOM(reducer, custom)                                                          \
  template <typename Scalar, typename... Properties, typename = std::enable_if_t<! Kokkos::Impl::is_specialization_of<Scalar, custom>::value>> \
  KOKKOS_DEDUCTION_GUIDE                                                                                             \
  reducer(Kokkos::View<custom<Scalar>, Properties...> const&)                                                        \
      -> reducer<Scalar, typename Kokkos::View<custom<Scalar>, Properties...>::memory_space>;                \
  template <typename Scalar, typename = std::enable_if_t<! Kokkos::is_view_v<Kokkos::Impl::remove_cvref_t<Scalar>>>> \
  KOKKOS_DEDUCTION_GUIDE reducer(custom<Scalar>&)                                                                    \
      -> reducer<Scalar>;

/// Specialization for backward compatibility, i.e. people that have hardcoded
/// @c Kokkos::Sum<int> or @c Kokkos::Sum<int,Kokkos::Cuda> for instance.
/// This is the user facing (public) API.
#define BACKWARD_COMPITABILITY_STRUCT(reducer)                           \
  /* When the target is a scalar. */                                     \
  template <typename ScalarType>                                         \
  struct reducer<ScalarType> : public Impl::reducer<                     \
    Impl::ReductionTarget<Kokkos::View<ScalarType, Kokkos::HostSpace>,   \
    Impl::ReductionTargetType::SCALAR>> {                                \
    using base_t = Impl::reducer<                                        \
      Impl::ReductionTarget<Kokkos::View<ScalarType, Kokkos::HostSpace>, \
      Impl::ReductionTargetType::SCALAR>>;                               \
    using base_t::base_t;                                                \
  };                                                                     \
  /* When the target is a view. */                                       \
  template <typename ScalarType, typename Space>                         \
  struct reducer<ScalarType, Space> : public Impl::reducer<              \
    Impl::ReductionTarget<Kokkos::View<ScalarType, Space>,               \
    Impl::ReductionTargetType::VIEW>> {                                  \
    using base_t = Impl::reducer<                                        \
      Impl::ReductionTarget<Kokkos::View<ScalarType, Space>,             \
      Impl::ReductionTargetType::VIEW>>;                                 \
    using base_t::base_t;                                                \
  };

#define BACKWARD_COMPITABILITY_STRUCT_CUSTOM(reducer, custom)                    \
  /* When the target is a scalar. */                                             \
  template <typename ScalarType>                                                 \
  struct reducer<ScalarType> : public Impl::reducer<                             \
    Impl::ReductionTarget<Kokkos::View<custom<ScalarType>, Kokkos::HostSpace>,   \
    Impl::ReductionTargetType::SCALAR>> {                                        \
    using base_t = Impl::reducer<                                                \
      Impl::ReductionTarget<Kokkos::View<custom<ScalarType>, Kokkos::HostSpace>, \
      Impl::ReductionTargetType::SCALAR>>;                                       \
    using base_t::base_t;                                                        \
  };                                                                             \
  /* When the target is a view. */                                               \
  template <typename ScalarType, typename Space>                                 \
  struct reducer<ScalarType, Space> : public Impl::reducer<                      \
    Impl::ReductionTarget<Kokkos::View<custom<ScalarType>, Space>,               \
    Impl::ReductionTargetType::VIEW>> {                                          \
    using base_t = Impl::reducer<                                                \
      Impl::ReductionTarget<Kokkos::View<custom<ScalarType>, Space>,             \
      Impl::ReductionTargetType::VIEW>>;                                         \
    using base_t::base_t;                                                        \
  };

/// Define the forwarding constructor consistently.
/// All reducers now forward to the reduction target struct.
#define REDUCER_CONSTRUCTOR_FORWARD(_reducer_)               \
  template <typename T, typename = std::enable_if_t<         \
    !Kokkos::is_reducer_v<Kokkos::Impl::remove_cvref_t<T>>>> \
  KOKKOS_FUNCTION                                            \
  _reducer_(T&& target_) : target(std::forward<T>(target_)) {}

namespace Impl {
template <class TargetType>
struct Sum {
 public:
  // Required
  using reducer    = Sum<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(Sum)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const { dest += src; }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};

} // namespace Impl

BACKWARD_COMPITABILITY_STRUCT(Sum)
DEDUCTION_GUIDE_FOR_REDUCER(Sum)

namespace Impl {
template <class TargetType>
struct Prod {
 public:
  // Required
  using reducer    = Prod<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(Prod)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const { dest *= src; }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}

BACKWARD_COMPITABILITY_STRUCT(Prod)
DEDUCTION_GUIDE_FOR_REDUCER(Prod)

namespace Impl {
template <typename TargetType>
struct Min {
 public:
  // Required
  using reducer    = Min<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(Min)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src < dest) dest = src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}

BACKWARD_COMPITABILITY_STRUCT(Min)
DEDUCTION_GUIDE_FOR_REDUCER(Min)

namespace Impl {
template <typename TargetType>
struct Max {
 public:
  // Required
  using reducer    = Max<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(Max)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src > dest) dest = src;
  }

  // Required
  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}
BACKWARD_COMPITABILITY_STRUCT(Max)
DEDUCTION_GUIDE_FOR_REDUCER(Max)

namespace Impl {
template <class TargetType>
struct LAnd {
 public:
  // Required
  using reducer    = LAnd<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(LAnd)

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest && src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}

BACKWARD_COMPITABILITY_STRUCT(LAnd)
DEDUCTION_GUIDE_FOR_REDUCER(LAnd)

namespace Impl {
template <class TargetType>
struct LOr {
 public:
  // Required
  using reducer    = LOr<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(LOr)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest || src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}
BACKWARD_COMPITABILITY_STRUCT(LOr)
DEDUCTION_GUIDE_FOR_REDUCER(LOr)

namespace Impl {
template <class TargetType>
struct BAnd {
 public:
  // Required
  using reducer    = BAnd<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(BAnd)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest & src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}
BACKWARD_COMPITABILITY_STRUCT(BAnd)
DEDUCTION_GUIDE_FOR_REDUCER(BAnd)

namespace Impl {
template <class TargetType>
struct BOr {
 public:
  // Required
  using reducer    = BOr<TargetType>;
  using value_type = std::remove_cv_t<typename TargetType::value_type>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(BOr)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest | src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}

BACKWARD_COMPITABILITY_STRUCT(BOr)
DEDUCTION_GUIDE_FOR_REDUCER(BOr)

template <class Scalar, class Index>
struct ValLocScalar {
  Scalar val;
  Index loc;
};

template <class Scalar, class Index, class Space>
struct MinLoc {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  // Required
  using reducer    = MinLoc<Scalar, Index, Space>;
  using value_type = ValLocScalar<scalar_type, index_type>;

  using result_view_type = Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  MinLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MinLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.val < dest.val)
      dest = src;
    else if (src.val == dest.val &&
             dest.loc == reduction_identity<index_type>::min()) {
      dest.loc = src.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE
MinLoc(View<ValLocScalar<Scalar, Index>, Properties...> const&) -> MinLoc<
    Scalar, Index,
    typename View<ValLocScalar<Scalar, Index>, Properties...>::memory_space>;

template <class Scalar, class Index, class Space>
struct MaxLoc {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  // Required
  using reducer    = MaxLoc<Scalar, Index, Space>;
  using value_type = ValLocScalar<scalar_type, index_type>;

  using result_view_type = Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  MaxLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MaxLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.val > dest.val)
      dest = src;
    else if (src.val == dest.val &&
             dest.loc == reduction_identity<index_type>::min()) {
      dest.loc = src.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE
MaxLoc(View<ValLocScalar<Scalar, Index>, Properties...> const&) -> MaxLoc<
    Scalar, Index,
    typename View<ValLocScalar<Scalar, Index>, Properties...>::memory_space>;

template <class Scalar>
struct MinMaxScalar {
  using value_type = Scalar;
  Scalar min_val, max_val;
};

namespace Impl {
template <class TargetType>
struct MinMax {
 private:
  using scalar_type = std::remove_cv_t<typename TargetType::value_type::value_type>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  // Required
  using reducer    = MinMax<TargetType>;
  using value_type = typename TargetType::value_type;

  static_assert(Kokkos::Impl::is_specialization_of<value_type, MinMaxScalar>::value);

  using result_view_type = typename TargetType::view_type;

 private:
  TargetType target;

 public:
  REDUCER_CONSTRUCTOR_FORWARD(MinMax)

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = reduction_identity<scalar_type>::max();
    val.min_val = reduction_identity<scalar_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *target.data.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return target.data; }

  KOKKOS_INLINE_FUNCTION
  constexpr bool references_scalar() const { return TargetType::type == Impl::ReductionTargetType::SCALAR; }
};
}

BACKWARD_COMPITABILITY_STRUCT_CUSTOM(MinMax, MinMaxScalar)
DEDUCTION_GUIDE_FOR_REDUCER_CUSTOM(MinMax, MinMaxScalar)

template <class Scalar, class Index>
struct MinMaxLocScalar {
  Scalar min_val, max_val;
  Index min_loc, max_loc;
};

template <class Scalar, class Index, class Space>
struct MinMaxLoc {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);

 public:
  // Required
  using reducer    = MinMaxLoc<Scalar, Index, Space>;
  using value_type = MinMaxLocScalar<scalar_type, index_type>;

  using result_view_type = Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  MinMaxLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MinMaxLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    } else if (dest.min_val == src.min_val &&
               dest.min_loc == reduction_identity<index_type>::min()) {
      dest.min_loc = src.min_loc;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    } else if (dest.max_val == src.max_val &&
               dest.max_loc == reduction_identity<index_type>::min()) {
      dest.max_loc = src.max_loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = reduction_identity<scalar_type>::max();
    val.min_val = reduction_identity<scalar_type>::min();
    val.max_loc = reduction_identity<index_type>::min();
    val.min_loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMaxLoc(
    View<MinMaxLocScalar<Scalar, Index>, Properties...> const&)
    -> MinMaxLoc<Scalar, Index,
                 typename View<MinMaxLocScalar<Scalar, Index>,
                               Properties...>::memory_space>;

// --------------------------------------------------
// reducers added to support std algorithms
// --------------------------------------------------

//
// MaxFirstLoc
//
template <class Scalar, class Index, class Space>
struct MaxFirstLoc {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = MaxFirstLoc<Scalar, Index, Space>;
  using value_type = ::Kokkos::ValLocScalar<scalar_type, index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  MaxFirstLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MaxFirstLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (dest.val < src.val) {
      dest = src;
    } else if (!(src.val < dest.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MaxFirstLoc(
    View<ValLocScalar<Scalar, Index>, Properties...> const&)
    -> MaxFirstLoc<Scalar, Index,
                   typename View<ValLocScalar<Scalar, Index>,
                                 Properties...>::memory_space>;

//
// MaxFirstLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
template <class Scalar, class Index, class ComparatorType, class Space>
struct MaxFirstLocCustomComparator {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer =
      MaxFirstLocCustomComparator<Scalar, Index, ComparatorType, Space>;
  using value_type = ::Kokkos::ValLocScalar<scalar_type, index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;
  ComparatorType m_comp;

 public:
  KOKKOS_INLINE_FUNCTION
  MaxFirstLocCustomComparator(value_type& value_, ComparatorType comp_)
      : value(&value_), references_scalar_v(true), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  MaxFirstLocCustomComparator(const result_view_type& value_,
                              ComparatorType comp_)
      : value(value_), references_scalar_v(false), m_comp(comp_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (m_comp(dest.val, src.val)) {
      dest = src;
    } else if (!m_comp(src.val, dest.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename ComparatorType,
          typename... Properties>
KOKKOS_DEDUCTION_GUIDE MaxFirstLocCustomComparator(
    View<ValLocScalar<Scalar, Index>, Properties...> const&, ComparatorType)
    -> MaxFirstLocCustomComparator<Scalar, Index, ComparatorType,
                                   typename View<ValLocScalar<Scalar, Index>,
                                                 Properties...>::memory_space>;

//
// MinFirstLoc
//
template <class Scalar, class Index, class Space>
struct MinFirstLoc {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = MinFirstLoc<Scalar, Index, Space>;
  using value_type = ::Kokkos::ValLocScalar<scalar_type, index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  MinFirstLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MinFirstLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.val < dest.val) {
      dest = src;
    } else if (!(dest.val < src.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinFirstLoc(
    View<ValLocScalar<Scalar, Index>, Properties...> const&)
    -> MinFirstLoc<Scalar, Index,
                   typename View<ValLocScalar<Scalar, Index>,
                                 Properties...>::memory_space>;

//
// MinFirstLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
template <class Scalar, class Index, class ComparatorType, class Space>
struct MinFirstLocCustomComparator {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer =
      MinFirstLocCustomComparator<Scalar, Index, ComparatorType, Space>;
  using value_type = ::Kokkos::ValLocScalar<scalar_type, index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;
  ComparatorType m_comp;

 public:
  KOKKOS_INLINE_FUNCTION
  MinFirstLocCustomComparator(value_type& value_, ComparatorType comp_)
      : value(&value_), references_scalar_v(true), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  MinFirstLocCustomComparator(const result_view_type& value_,
                              ComparatorType comp_)
      : value(value_), references_scalar_v(false), m_comp(comp_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (m_comp(src.val, dest.val)) {
      dest = src;
    } else if (!m_comp(dest.val, src.val)) {
      dest.loc = (src.loc < dest.loc) ? src.loc : dest.loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename ComparatorType,
          typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinFirstLocCustomComparator(
    View<ValLocScalar<Scalar, Index>, Properties...> const&, ComparatorType)
    -> MinFirstLocCustomComparator<Scalar, Index, ComparatorType,
                                   typename View<ValLocScalar<Scalar, Index>,
                                                 Properties...>::memory_space>;

//
// MinMaxFirstLastLoc
//
template <class Scalar, class Index, class Space>
struct MinMaxFirstLastLoc {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = MinMaxFirstLastLoc<Scalar, Index, Space>;
  using value_type = ::Kokkos::MinMaxLocScalar<scalar_type, index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  MinMaxFirstLastLoc(value_type& value_)
      : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MinMaxFirstLastLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    } else if (!(dest.min_val < src.min_val)) {
      dest.min_loc = (src.min_loc < dest.min_loc) ? src.min_loc : dest.min_loc;
    }

    if (dest.max_val < src.max_val) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    } else if (!(src.max_val < dest.max_val)) {
      dest.max_loc = (src.max_loc > dest.max_loc) ? src.max_loc : dest.max_loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = ::Kokkos::reduction_identity<scalar_type>::max();
    val.min_val = ::Kokkos::reduction_identity<scalar_type>::min();
    val.max_loc = ::Kokkos::reduction_identity<index_type>::max();
    val.min_loc = ::Kokkos::reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMaxFirstLastLoc(
    View<MinMaxLocScalar<Scalar, Index>, Properties...> const&)
    -> MinMaxFirstLastLoc<Scalar, Index,
                          typename View<MinMaxLocScalar<Scalar, Index>,
                                        Properties...>::memory_space>;

//
// MinMaxFirstLastLocCustomComparator
// recall that comp(a,b) returns true is a < b
//
template <class Scalar, class Index, class ComparatorType, class Space>
struct MinMaxFirstLastLocCustomComparator {
 private:
  using scalar_type = std::remove_cv_t<Scalar>;
  using index_type  = std::remove_cv_t<Index>;
  static_assert(!std::is_pointer_v<scalar_type> &&
                !std::is_array_v<scalar_type>);
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer =
      MinMaxFirstLastLocCustomComparator<Scalar, Index, ComparatorType, Space>;
  using value_type = ::Kokkos::MinMaxLocScalar<scalar_type, index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;
  ComparatorType m_comp;

 public:
  KOKKOS_INLINE_FUNCTION
  MinMaxFirstLastLocCustomComparator(value_type& value_, ComparatorType comp_)
      : value(&value_), references_scalar_v(true), m_comp(comp_) {}

  KOKKOS_INLINE_FUNCTION
  MinMaxFirstLastLocCustomComparator(const result_view_type& value_,
                                     ComparatorType comp_)
      : value(value_), references_scalar_v(false), m_comp(comp_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (m_comp(src.min_val, dest.min_val)) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    } else if (!m_comp(dest.min_val, src.min_val)) {
      dest.min_loc = (src.min_loc < dest.min_loc) ? src.min_loc : dest.min_loc;
    }

    if (m_comp(dest.max_val, src.max_val)) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    } else if (!m_comp(src.max_val, dest.max_val)) {
      dest.max_loc = (src.max_loc > dest.max_loc) ? src.max_loc : dest.max_loc;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_val = ::Kokkos::reduction_identity<scalar_type>::max();
    val.min_val = ::Kokkos::reduction_identity<scalar_type>::min();
    val.max_loc = ::Kokkos::reduction_identity<index_type>::max();
    val.min_loc = ::Kokkos::reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Scalar, typename Index, typename ComparatorType,
          typename... Properties>
KOKKOS_DEDUCTION_GUIDE MinMaxFirstLastLocCustomComparator(
    View<MinMaxLocScalar<Scalar, Index>, Properties...> const&, ComparatorType)
    -> MinMaxFirstLastLocCustomComparator<
        Scalar, Index, ComparatorType,
        typename View<MinMaxLocScalar<Scalar, Index>,
                      Properties...>::memory_space>;

//
// FirstLoc
//
template <class Index>
struct FirstLocScalar {
  Index min_loc_true;
};

template <class Index, class Space>
struct FirstLoc {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = FirstLoc<Index, Space>;
  using value_type = FirstLocScalar<index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  FirstLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  FirstLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.min_loc_true = (src.min_loc_true < dest.min_loc_true)
                            ? src.min_loc_true
                            : dest.min_loc_true;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.min_loc_true = ::Kokkos::reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE
FirstLoc(View<FirstLocScalar<Index>, Properties...> const&) -> FirstLoc<
    Index, typename View<FirstLocScalar<Index>, Properties...>::memory_space>;

//
// LastLoc
//
template <class Index>
struct LastLocScalar {
  Index max_loc_true;
};

template <class Index, class Space>
struct LastLoc {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = LastLoc<Index, Space>;
  using value_type = LastLocScalar<index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  LastLoc(value_type& value_) : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  LastLoc(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.max_loc_true = (src.max_loc_true > dest.max_loc_true)
                            ? src.max_loc_true
                            : dest.max_loc_true;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_loc_true = ::Kokkos::reduction_identity<index_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE LastLoc(View<LastLocScalar<Index>, Properties...> const&)
    -> LastLoc<Index, typename View<LastLocScalar<Index>,
                                    Properties...>::memory_space>;

template <class Index>
struct StdIsPartScalar {
  Index max_loc_true, min_loc_false;
};

//
// StdIsPartitioned
//
template <class Index, class Space>
struct StdIsPartitioned {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = StdIsPartitioned<Index, Space>;
  using value_type = StdIsPartScalar<index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  StdIsPartitioned(value_type& value_)
      : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  StdIsPartitioned(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.max_loc_true = (dest.max_loc_true < src.max_loc_true)
                            ? src.max_loc_true
                            : dest.max_loc_true;

    dest.min_loc_false = (dest.min_loc_false < src.min_loc_false)
                             ? dest.min_loc_false
                             : src.min_loc_false;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.max_loc_true  = ::Kokkos::reduction_identity<index_type>::max();
    val.min_loc_false = ::Kokkos::reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE StdIsPartitioned(
    View<StdIsPartScalar<Index>, Properties...> const&)
    -> StdIsPartitioned<Index, typename View<StdIsPartScalar<Index>,
                                             Properties...>::memory_space>;

template <class Index>
struct StdPartPointScalar {
  Index min_loc_false;
};

//
// StdPartitionPoint
//
template <class Index, class Space>
struct StdPartitionPoint {
 private:
  using index_type = std::remove_cv_t<Index>;
  static_assert(std::is_integral_v<index_type>);

 public:
  // Required
  using reducer    = StdPartitionPoint<Index, Space>;
  using value_type = StdPartPointScalar<index_type>;

  using result_view_type = ::Kokkos::View<value_type, Space>;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  StdPartitionPoint(value_type& value_)
      : value(&value_), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  StdPartitionPoint(const result_view_type& value_)
      : value(value_), references_scalar_v(false) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.min_loc_false = (dest.min_loc_false < src.min_loc_false)
                             ? dest.min_loc_false
                             : src.min_loc_false;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val.min_loc_false = ::Kokkos::reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return references_scalar_v; }
};

template <typename Index, typename... Properties>
KOKKOS_DEDUCTION_GUIDE StdPartitionPoint(
    View<StdPartPointScalar<Index>, Properties...> const&)
    -> StdPartitionPoint<Index, typename View<StdPartPointScalar<Index>,
                                              Properties...>::memory_space>;

}  // namespace Kokkos
namespace Kokkos {
namespace Impl {

template <typename FunctorType, typename FunctorAnalysisReducerType,
          typename Enable>
class CombinedFunctorReducer {
 public:
  using functor_type = FunctorType;
  using reducer_type = FunctorAnalysisReducerType;
  CombinedFunctorReducer(const FunctorType& functor,
                         const FunctorAnalysisReducerType& reducer)
      : m_functor(functor), m_reducer(reducer) {}
  KOKKOS_FUNCTION const FunctorType& get_functor() const { return m_functor; }
  KOKKOS_FUNCTION const FunctorAnalysisReducerType& get_reducer() const {
    return m_reducer;
  }

 private:
  FunctorType m_functor;
  FunctorAnalysisReducerType m_reducer;
};
template <typename FunctorType, typename FunctorAnalysisReducerType>
class CombinedFunctorReducer<
    FunctorType, FunctorAnalysisReducerType,
    std::enable_if_t<std::is_same_v<
        FunctorType, typename FunctorAnalysisReducerType::functor_type>>> {
 public:
  using functor_type = FunctorType;
  using reducer_type = FunctorAnalysisReducerType;
  CombinedFunctorReducer(const FunctorType& functor,
                         const FunctorAnalysisReducerType&)
      : m_reducer(functor) {}
  KOKKOS_FUNCTION const FunctorType& get_functor() const {
    return m_reducer.get_functor();
  }
  KOKKOS_FUNCTION const FunctorAnalysisReducerType& get_reducer() const {
    return m_reducer;
  }

 private:
  FunctorAnalysisReducerType m_reducer;
};

template <class T, class ReturnType, class ValueTraits>
struct ParallelReduceReturnValue;

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<Kokkos::is_view<ReturnType>::value>, ReturnType,
    FunctorType> {
  using return_type  = ReturnType;
  using reducer_type = InvalidType;

  using value_type_scalar = typename return_type::value_type;
  using value_type_array  = typename return_type::value_type* const;

  using value_type = std::conditional_t<return_type::rank == 0,
                                        value_type_scalar, value_type_array>;

  static return_type& return_value(ReturnType& return_val, const FunctorType&) {
    return return_val;
  }
};

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<!Kokkos::is_view<ReturnType>::value &&
                     (!std::is_array_v<ReturnType> &&
                      !std::is_pointer_v<
                          ReturnType>)&&!Kokkos::is_reducer<ReturnType>::value>,
    ReturnType, FunctorType> {
  using return_type =
      Kokkos::View<ReturnType, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  using reducer_type = InvalidType;

  using value_type = typename return_type::value_type;

  static return_type return_value(ReturnType& return_val, const FunctorType&) {
    return return_type(&return_val);
  }
};

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<(std::is_array_v<ReturnType> ||
                      std::is_pointer_v<ReturnType>)>,
    ReturnType, FunctorType> {
  using return_type = Kokkos::View<std::remove_const_t<ReturnType>,
                                   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

  using reducer_type = InvalidType;

  using value_type = typename return_type::value_type[];

  static return_type return_value(ReturnType& return_val,
                                  const FunctorType& functor) {
    if (std::is_array_v<ReturnType>)
      return return_type(return_val);
    else
      return return_type(return_val, functor.value_count);
  }
};

template <class ReturnType, class FunctorType>
struct ParallelReduceReturnValue<
    std::enable_if_t<Kokkos::is_reducer<ReturnType>::value>, ReturnType,
    FunctorType> {
  using return_type  = typename ReturnType::result_view_type;
  using reducer_type = ReturnType;
  using value_type   = typename return_type::value_type;

  static auto return_value(ReturnType& return_val, const FunctorType&) {
    return return_val.view();
  }
};

template <class T, class ReturnType, class FunctorType>
struct ParallelReducePolicyType;

template <class PolicyType, class FunctorType>
struct ParallelReducePolicyType<
    std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value>,
    PolicyType, FunctorType> {
  using policy_type = PolicyType;
  static PolicyType policy(const PolicyType& policy_) { return policy_; }
};

template <class PolicyType, class FunctorType>
struct ParallelReducePolicyType<
    std::enable_if_t<std::is_integral_v<PolicyType>>, PolicyType, FunctorType> {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;

  using policy_type = Kokkos::RangePolicy<execution_space>;

  static policy_type policy(const PolicyType& policy_) {
    return policy_type(0, policy_);
  }
};

template <class FunctorType, class ExecPolicy, class ValueType,
          class ExecutionSpace>
struct ParallelReduceFunctorType {
  using functor_type = FunctorType;
  static const functor_type& functor(const functor_type& functor) {
    return functor;
  }
};

template <class PolicyType, class FunctorType, class ReturnType>
struct ParallelReduceAdaptor {
  using return_value_adapter =
      Impl::ParallelReduceReturnValue<void, ReturnType, FunctorType>;

  static inline void execute_impl(const std::string& label,
                                  const PolicyType& policy,
                                  const FunctorType& functor,
                                  ReturnType& return_value) {
    using PassedReducerType = typename return_value_adapter::reducer_type;
    uint64_t kpID           = 0;

    using ReducerSelector =
        Kokkos::Impl::if_c<std::is_same_v<InvalidType, PassedReducerType>,
                           FunctorType, PassedReducerType>;
    using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                     PolicyType, typename ReducerSelector::type,
                                     typename return_value_adapter::value_type>;
    using CombinedFunctorReducerType =
        CombinedFunctorReducer<FunctorType, typename Analysis::Reducer>;

    CombinedFunctorReducerType functor_reducer(
        functor, typename Analysis::Reducer(
                     ReducerSelector::select(functor, return_value)));
    const auto& response = Kokkos::Tools::Impl::begin_parallel_reduce<
        typename return_value_adapter::reducer_type>(policy, functor_reducer,
                                                     label, kpID);
    const auto& inner_policy = response.policy;

    auto closure = construct_with_shared_allocation_tracking_disabled<
        Impl::ParallelReduce<CombinedFunctorReducerType, PolicyType,
                             typename Impl::FunctorPolicyExecutionSpace<
                                 FunctorType, PolicyType>::execution_space>>(
        functor_reducer, inner_policy,
        return_value_adapter::return_value(return_value, functor));
    closure.execute();

    Kokkos::Tools::Impl::end_parallel_reduce<PassedReducerType>(
        inner_policy, functor, label, kpID);
  }

  static constexpr bool is_array_reduction =
      Impl::FunctorAnalysis<
          Impl::FunctorPatternInterface::REDUCE, PolicyType, FunctorType,
          typename return_value_adapter::value_type>::StaticValueSize == 0;

  template <typename Dummy = ReturnType>
  static inline std::enable_if_t<!(is_array_reduction &&
                                   std::is_pointer_v<Dummy>)>
  execute(const std::string& label, const PolicyType& policy,
          const FunctorType& functor, ReturnType& return_value) {
    execute_impl(label, policy, functor, return_value);
  }
};
}  // namespace Impl

//----------------------------------------------------------------------------

/*! \fn void parallel_reduce(label,policy,functor,return_argument)
    \brief Perform a parallel reduction.
    \param label An optional Label giving the call name. Must be able to
   construct a std::string from the argument. \param policy A Kokkos Execution
   Policy, such as an integer, a RangePolicy or a TeamPolicy. \param functor A
   functor with a reduction operator, and optional init, join and final
   functions. \param return_argument A return argument which can be a scalar, a
   View, or a ReducerStruct. This argument can be left out if the functor has a
   final function.
*/

// Parallel Reduce Blocking behavior

namespace Impl {
template <typename T>
struct ReducerHasTestReferenceFunction {
  template <typename E>
  static std::true_type test_func(decltype(&E::references_scalar));
  template <typename E>
  static std::false_type test_func(...);

  enum {
    value = std::is_same_v<std::true_type, decltype(test_func<T>(nullptr))>
  };
};

template <class ExecutionSpace, class Arg>
constexpr std::enable_if_t<
    // constraints only necessary because SFINAE lacks subsumption
    !ReducerHasTestReferenceFunction<Arg>::value &&
        !Kokkos::is_view<Arg>::value,
    // return type:
    bool>
parallel_reduce_needs_fence(ExecutionSpace const&, Arg const&) {
  return true;
}

template <class ExecutionSpace, class Reducer>
constexpr std::enable_if_t<
    // equivalent to:
    // (requires (Reducer const& r) {
    //   { reducer.references_scalar() } -> std::convertible_to<bool>;
    // })
    ReducerHasTestReferenceFunction<Reducer>::value,
    // return type:
    bool>
parallel_reduce_needs_fence(ExecutionSpace const&, Reducer const& reducer) {
  return reducer.references_scalar();
}

template <class ExecutionSpace, class ViewLike>
constexpr std::enable_if_t<
    // requires Kokkos::ViewLike<ViewLike>
    Kokkos::is_view<ViewLike>::value,
    // return type:
    bool>
parallel_reduce_needs_fence(ExecutionSpace const&, ViewLike const&) {
  return false;
}

template <class ExecutionSpace, class... Args>
struct ParallelReduceFence {
  template <class... ArgsDeduced>
  static void fence(const ExecutionSpace& ex, const std::string& name,
                    ArgsDeduced&&... args) {
    if (Impl::parallel_reduce_needs_fence(ex, (ArgsDeduced&&)args...)) {
      ex.fence(name);
    }
  }
};

}  // namespace Impl

/** \brief  Parallel reduction
 *
 * parallel_reduce performs parallel reductions with arbitrary functions - i.e.
 * it is not solely data based. The call expects up to 4 arguments:
 *
 *
 * Example of a parallel_reduce functor for a POD (plain old data) value type:
 * \code
 *  class FunctorType { // For POD value type
 *  public:
 *    using execution_space = ...;
 *    using value_type = <podType>;
 *    void operator()( <intType> iwork , <podType> & update ) const ;
 *    void init( <podType> & update ) const ;
 *    void join(       <podType> & update ,
 *               const <podType> & input ) const ;
 *
 *    void final( <podType> & update ) const ;
 *  };
 * \endcode
 *
 * Example of a parallel_reduce functor for an array of POD (plain old data)
 * values:
 * \code
 *  class FunctorType { // For array of POD value
 *  public:
 *    using execution_space = ...;
 *    using value_type = <podType>[];
 *    void operator()( <intType> , <podType> update[] ) const ;
 *    void init( <podType> update[] ) const ;
 *    void join(       <podType> update[] ,
 *               const <podType> input[] ) const ;
 *
 *    void final( <podType> update[] ) const ;
 *  };
 * \endcode
 */

// ReturnValue is scalar or array: take by reference

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        !(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const std::string& label, const PolicyType& policy,
                const FunctorType& functor, ReturnType& return_value) {
  static_assert(
      !std::is_const_v<ReturnType>,
      "A const reduction result type is only allowed for a View, pointer or "
      "reducer return type!");

  Impl::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
      label, policy, functor, return_value);
  Impl::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
      fence(
          policy.space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        !(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const PolicyType& policy, const FunctorType& functor,
                ReturnType& return_value) {
  static_assert(
      !std::is_const_v<ReturnType>,
      "A const reduction result type is only allowed for a View, pointer or "
      "reducer return type!");

  Impl::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
      "", policy, functor, return_value);
  Impl::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
      fence(
          policy.space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const size_t& policy, const FunctorType& functor,
                ReturnType& return_value) {
  static_assert(
      !std::is_const_v<ReturnType>,
      "A const reduction result type is only allowed for a View, pointer or "
      "reducer return type!");

  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;

  Impl::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
      "", policy_type(0, policy), functor, return_value);
  Impl::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
      fence(
          typename policy_type::execution_space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
parallel_reduce(const std::string& label, const size_t& policy,
                const FunctorType& functor, ReturnType& return_value) {
  static_assert(
      !std::is_const_v<ReturnType>,
      "A const reduction result type is only allowed for a View, pointer or "
      "reducer return type!");

  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;
  Impl::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
      label, policy_type(0, policy), functor, return_value);
  Impl::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
      fence(
          typename policy_type::execution_space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

// ReturnValue as View or Reducer: take by copy to allow for inline construction

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        (Kokkos::is_view<ReturnType>::value ||
                         Kokkos::is_reducer<ReturnType>::value ||
                         std::is_pointer_v<ReturnType>)>
parallel_reduce(const std::string& label, const PolicyType& policy,
                const FunctorType& functor, const ReturnType& return_value) {
  ReturnType return_value_impl = return_value;
  Impl::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
      label, policy, functor, return_value_impl);
  Impl::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
      fence(
          policy.space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class PolicyType, class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value &&
                        (Kokkos::is_view<ReturnType>::value ||
                         Kokkos::is_reducer<ReturnType>::value ||
                         std::is_pointer_v<ReturnType>)>
parallel_reduce(const PolicyType& policy, const FunctorType& functor,
                const ReturnType& return_value) {
  ReturnType return_value_impl = return_value;
  Impl::ParallelReduceAdaptor<PolicyType, FunctorType, ReturnType>::execute(
      "", policy, functor, return_value_impl);
  Impl::ParallelReduceFence<typename PolicyType::execution_space, ReturnType>::
      fence(
          policy.space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_view<ReturnType>::value ||
                        Kokkos::is_reducer<ReturnType>::value ||
                        std::is_pointer_v<ReturnType>>
parallel_reduce(const size_t& policy, const FunctorType& functor,
                const ReturnType& return_value) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;
  ReturnType return_value_impl = return_value;
  Impl::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
      "", policy_type(0, policy), functor, return_value_impl);
  Impl::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
      fence(
          typename policy_type::execution_space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<Kokkos::is_view<ReturnType>::value ||
                        Kokkos::is_reducer<ReturnType>::value ||
                        std::is_pointer_v<ReturnType>>
parallel_reduce(const std::string& label, const size_t& policy,
                const FunctorType& functor, const ReturnType& return_value) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;
  ReturnType return_value_impl = return_value;
  Impl::ParallelReduceAdaptor<policy_type, FunctorType, ReturnType>::execute(
      label, policy_type(0, policy), functor, return_value_impl);
  Impl::ParallelReduceFence<typename policy_type::execution_space, ReturnType>::
      fence(
          typename policy_type::execution_space(),
          "Kokkos::parallel_reduce: fence due to result being value, not view",
          return_value);
}

// No Return Argument

template <class PolicyType, class FunctorType>
inline void parallel_reduce(
    const std::string& label, const PolicyType& policy,
    const FunctorType& functor,
    std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value>* =
        nullptr) {
  using FunctorAnalysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE, PolicyType,
                            FunctorType, void>;
  using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                                        typename FunctorAnalysis::value_type,
                                        typename FunctorAnalysis::pointer_type>;

  static_assert(
      FunctorAnalysis::has_final_member_function,
      "Calling parallel_reduce without either return value or final function.");

  using result_view_type =
      Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  result_view_type result_view;

  Impl::ParallelReduceAdaptor<PolicyType, FunctorType,
                              result_view_type>::execute(label, policy, functor,
                                                         result_view);
}

template <class PolicyType, class FunctorType>
inline void parallel_reduce(
    const PolicyType& policy, const FunctorType& functor,
    std::enable_if_t<Kokkos::is_execution_policy<PolicyType>::value>* =
        nullptr) {
  using FunctorAnalysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE, PolicyType,
                            FunctorType, void>;
  using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                                        typename FunctorAnalysis::value_type,
                                        typename FunctorAnalysis::pointer_type>;

  static_assert(
      FunctorAnalysis::has_final_member_function,
      "Calling parallel_reduce without either return value or final function.");

  using result_view_type =
      Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  result_view_type result_view;

  Impl::ParallelReduceAdaptor<PolicyType, FunctorType,
                              result_view_type>::execute("", policy, functor,
                                                         result_view);
}

template <class FunctorType>
inline void parallel_reduce(const size_t& policy, const FunctorType& functor) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;
  using FunctorAnalysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE, policy_type,
                            FunctorType, void>;
  using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                                        typename FunctorAnalysis::value_type,
                                        typename FunctorAnalysis::pointer_type>;

  static_assert(
      FunctorAnalysis::has_final_member_function,
      "Calling parallel_reduce without either return value or final function.");

  using result_view_type =
      Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  result_view_type result_view;

  Impl::ParallelReduceAdaptor<policy_type, FunctorType,
                              result_view_type>::execute("",
                                                         policy_type(0, policy),
                                                         functor, result_view);
}

template <class FunctorType>
inline void parallel_reduce(const std::string& label, const size_t& policy,
                            const FunctorType& functor) {
  using policy_type =
      typename Impl::ParallelReducePolicyType<void, size_t,
                                              FunctorType>::policy_type;
  using FunctorAnalysis =
      Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE, policy_type,
                            FunctorType, void>;
  using value_type = std::conditional_t<(FunctorAnalysis::StaticValueSize != 0),
                                        typename FunctorAnalysis::value_type,
                                        typename FunctorAnalysis::pointer_type>;

  static_assert(
      FunctorAnalysis::has_final_member_function,
      "Calling parallel_reduce without either return value or final function.");

  using result_view_type =
      Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  result_view_type result_view;

  Impl::ParallelReduceAdaptor<policy_type, FunctorType,
                              result_view_type>::execute(label,
                                                         policy_type(0, policy),
                                                         functor, result_view);
}

}  // namespace Kokkos

#endif  // KOKKOS_PARALLEL_REDUCE_HPP
