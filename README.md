# LSTM-tensorflow
Implementation of an LSTMN, a training loop, and generation of new data using the LSTMN

**Hyperparmeters.py:** contains all the network parameters and the learning parameters.
  * ratio_test: amount of data for the test set, the remainder is used for training,
  * seq_size: size of sequences used both for training and testing,
  * num_hidden: number of hidden neurons,
  * batch_size: batch size, 0 means full-batch training,
  * epoch: number of epochs to train for.

**PredictSales.py:** contains the implementation itself (this is a preliminary work for the Kaggle competition 'predict future sales').
  * Creation of a noised sinusoid as data for testing the model.
  * Construction of the LSTMN tensorflow graph and execution of a training loop
  * Function pred_fun generates a prediction of length length_of_story using starting_data and the LSTMN model stored in "/tmp/model.ckpt".
