#include <mlpack/prereqs.hpp>

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

  virtual double Loss() { return 0; }

  virtual size_t WeightSize() { return 0; }

  virtual size_t OutputWidth() { return 0; }

  virtual size_t OutputHeight() { return 0; }

};

}
}
