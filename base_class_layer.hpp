#include <mlpack/prereqs.hpp>
#include <iostream>

namespace mlpack {
namespace ann {

template<typename eT>
class Layer
{
 public:

  virtual void Reset()
  {
    std::cout<<"Reset in Layer";
  }

  virtual void Forward(const arma::Mat<eT>&,
                       arma::Mat<eT>&)
  {
    std::cout<<"Forward in Layer";
  }

  virtual void Backward(const arma::Mat<eT>&,
                const arma::Mat<eT>&,
                arma::Mat<eT>&)
  {
    std::cout<<"Backward in Layer";
  }

  virtual void Gradient(const arma::Mat<eT>&,
                const arma::Mat<eT>&,
                arma::Mat<eT>&)
  {
    std::cout<<"Gradient in Layer";
  }

};

}
}
