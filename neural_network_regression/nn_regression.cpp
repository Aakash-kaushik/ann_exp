#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include "../linear.hpp"
#include <mlpack/core/data/split_data.hpp>
#include "../ffn.hpp"
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <ensmallen.hpp>
#include <chrono>
#include "../linear.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;


double MSE(arma::mat& pred, arma::mat& Y)
{
  return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

int main()
{
  //! - H1: The number of neurons in the 1st layer.
  constexpr size_t H1 = 64;
  //! - H2: The number of neurons in the 2nd layer.
  constexpr size_t H2 = 128;
  //! - H3: The number of neurons in the 3rd layer.
  constexpr size_t H3 = 64;


  // Number of epochs for training.
  const int EPOCHS = 100;
  //! - STEP_SIZE: Step size of the optimizer.
  constexpr double STEP_SIZE = 5e-2;
  //! - BATCH_SIZE: Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 32;
  //! - STOP_TOLERANCE: Stop tolerance;
  // A very small number implies that we do all iterations.
  constexpr double STOP_TOLERANCE = 1e-8;

  // In Armadillo rows represent features, columns represent data points.
  std::cout << "Random init data." << std::endl;
  // If dataset is not loaded correctly, exit.

  arma::mat dataset;
  dataset.randn(65,400);

  // Split the dataset into training and validation sets.
  arma::mat trainData, validData;
  data::Split(dataset, trainData, validData, 0.1);

  std::cout<<"trainData size: "<<arma::size(trainData)<<"\t"<<"validData size "<<arma::size(validData);

  // The train and valid datasets contain both - the features as well as the
  // prediction. Split these into separate matrices.
  arma::mat trainX = trainData.submat(1, 0, trainData.n_rows - 1, trainData.n_cols - 1);
  arma::mat validX = validData.submat(1, 0, validData.n_rows - 1, validData.n_cols - 1);

  // Create prediction data for training and validatiion datasets.
  arma::mat trainY = trainData.row(0);
  arma::mat validY = validData.row(0);

  auto start = std::chrono::steady_clock::now();
  FFN<MeanSquaredError<>, HeInitialization> model;
  model.Add<Linear<>>(size_t(64), H1);
  model.Add<Linear<>>(H1, H2);
  model.Add<Linear<>>(H2, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, H3);
  model.Add<Linear<>>(H3, 1);

  // Set parameters for the Stochastic Gradient Descent (SGD) optimizer.
  ens::Adam optimizer(
      STEP_SIZE,  // Step size of the optimizer.
      BATCH_SIZE, // Batch size. Number of data points that are used in each
                  // iteration.
      0.9,        // Exponential decay rate for the first moment estimates.
      0.999,      // Exponential decay rate for the weighted infinity norm
                  // estimates.
      1e-8, // Value used to initialise the mean squared gradient parameter.
      trainData.n_cols * EPOCHS, // Max number of iterations.
      STOP_TOLERANCE,            // Tolerance.
      true);

  model.Train(trainX,
              trainY,
              optimizer,
              // PrintLoss Callback prints loss for each epoch.
              ens::PrintLoss(),
              // Progressbar Callback prints progress bar for each epoch.
              ens::ProgressBar()
              // Stops the optimization process if the loss stops decreasing
              // or no improvement has been made. This will terminate the
              // optimization once we obtain a minima on training set.
              );

  std::cout << "Finished training."<< std::endl;

  arma::mat predOut;
  model.Predict(validX, predOut);

  auto end = std::chrono::steady_clock::now();
  double exec_time = (end - start).count();

  double validMSE = MSE(validY, predOut);

  std::cout << "Mean Squared Error on Prediction data points: " << validMSE
            << std::endl;

  std::cout<<"Time taken to execute: "<<exec_time<<std::endl;
  return 0;
}
