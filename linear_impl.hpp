/**
 * @file methods/ann/layer/linear_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Linear layer class also known as fully-connected layer
 * or affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP

// In case it hasn't yet been included.
#include "linear.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear<InputDataType, OutputDataType, RegularizerType>::Linear() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear<InputDataType, OutputDataType, RegularizerType>::Linear(
    const size_t inSize,
    const size_t outSize,
    RegularizerType regularizer) :
    inSize(inSize),
    outSize(outSize),
    regularizer(regularizer)
{
  weights.set_size(WeightSize(), 1);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear<InputDataType, OutputDataType, RegularizerType>::Linear(
    const Linear& layer) :
    inSize(layer.inSize),
    outSize(layer.outSize),
    weights(layer.weights),
    regularizer(layer.regularizer)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear<InputDataType, OutputDataType, RegularizerType>::Linear(
    Linear&& layer) :
    inSize(0),
    outSize(0),
    weights(std::move(layer.weights)),
    regularizer(std::move(layer.regularizer))
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear<InputDataType, OutputDataType, RegularizerType>&
Linear<InputDataType, OutputDataType, RegularizerType>::
operator=(const Linear& layer)
{
  if (this != &layer)
  {
    inSize = layer.inSize;
    outSize = layer.outSize;
    weights = layer.weights;
    regularizer = layer.regularizer;
  }
  return *this;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear<InputDataType, OutputDataType, RegularizerType>&
Linear<InputDataType, OutputDataType, RegularizerType>::
operator=(Linear&& layer)
{
  if (this != &layer)
  {
    inSize = layer.inSize;
    outSize = layer.outSize;
    weights = std::move(layer.weights);
    regularizer = std::move(layer.regularizer);
  }
  return *this;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void Linear<InputDataType, OutputDataType, RegularizerType>::Reset()
{
  weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
  bias = arma::mat(weights.memptr() + weight.n_elem,
      outSize, 1, false, false);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void Linear<InputDataType, OutputDataType, RegularizerType>::Forward(
    const InputDataType& input, OutputDataType& output)
{
  output = weight * input;
  output.each_col() += bias;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void Linear<InputDataType, OutputDataType, RegularizerType>::Backward(
    const InputDataType& /* input */, const InputDataType& gy, OutputDataType& g)
{
  g = weight.t() * gy;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void Linear<InputDataType, OutputDataType, RegularizerType>::Gradient(
    const InputDataType& input,
    const InputDataType& error,
    OutputDataType& gradient)
{
  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      error * input.t());
  gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(error, 1);
  regularizer.Evaluate(weights, gradient);
}

//template<typename InputDataType, typename OutputDataType,
//    typename RegularizerType>
//template<typename Archive>
//void Linear<InputDataType, OutputDataType, RegularizerType>::serialize(
//    Archive& ar, const uint32_t /* version */)
//{
//  ar(CEREAL_NVP(inSize));
//  ar(CEREAL_NVP(outSize));
//  ar(CEREAL_NVP(weights));
//}

} // namespace ann
} // namespace mlpack

#endif
