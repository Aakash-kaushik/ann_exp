#include <mlpack/prereqs.hpp>
#include <iostream>

namespace mlpack {
namespace ann {

template<typename InputDataType,
         typename OutputDataType>
class Layer
{
 public:

  virtual ~Layer(){}

  virtual void Reset()=0;

  virtual void Forward(const InputDataType&,
                       OutputDataType&)=0;

  virtual void Backward(const InputDataType&,
                        const InputDataType&,
                        OutputDataType&)=0;
 
  virtual void Gradient(const InputDataType&,
                        const InputDataType&,
                        OutputDataType&)=0;

  virtual OutputDataType const& OutputParameter() const=0;

  virtual OutputDataType& OutputParameter()=0;

  virtual OutputDataType const& Delta() const=0;
  
  virtual OutputDataType& Delta()=0;

  



  
};

}
}
