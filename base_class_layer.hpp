#include <mlpack/prereqs.hpp>
#include <iostream>

namespace mlpack {
namespace ann {

template<typename InputDataType,
         typename OutputDataType>
class Layer
{
 public:

  virtual void Reset()
  {
    std::cout<<"Reset in Layer";
  }

  virtual void Forward(const InputDataType&,
                       OutputDataType&)
  {
    std::cout<<"Forward in Layer";
  }

  virtual void Backward(const InputDataType&,
                const InputDataType&,
                OutputDataType&)
  {
    std::cout<<"Backward in Layer";
  }

  virtual void Gradient(const InputDataType&,
                const InputDataType&,
                OutputDataType&)
  {
    std::cout<<"Gradient in Layer";
  }

};

}
}
