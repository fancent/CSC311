import numpy as np

def load_data(filename, load2=True, load3=True):
  """Loads data for 2's and 3's
  Inputs:
    filename: Name of the file.
    load2: If True, load data for 2's.
    load3: If True, load data for 3's.
  """
  assert (load2 or load3), "Atleast one dataset must be loaded."
  data = np.load(filename)
  if load2 and load3:
    inputs_train = np.hstack((data['train2'], data['train3']))
    inputs_valid = np.hstack((data['valid2'], data['valid3']))
    inputs_test = np.hstack((data['test2'], data['test3']))
    target_train = np.hstack((np.zeros((1, data['train2'].shape[1])), np.ones((1, data['train3'].shape[1]))))
    target_valid = np.hstack((np.zeros((1, data['valid2'].shape[1])), np.ones((1, data['valid3'].shape[1]))))
    target_test = np.hstack((np.zeros((1, data['test2'].shape[1])), np.ones((1, data['test3'].shape[1]))))
  else:
    if load2:
      inputs_train = data['train2']
      target_train = np.zeros((1, data['train2'].shape[1]))
      inputs_valid = data['valid2']
      target_valid = np.zeros((1, data['valid2'].shape[1]))
      inputs_test = data['test2']
      target_test = np.zeros((1, data['test2'].shape[1]))
    else:
      inputs_train = data['train3']
      target_train = np.zeros((1, data['train3'].shape[1]))
      inputs_valid = data['valid3']
      target_valid = np.zeros((1, data['valid3'].shape[1]))
      inputs_test = data['test3']
      target_test = np.zeros((1, data['test3'].shape[1]))

  return inputs_train.T, inputs_valid.T, inputs_test.T, target_train.T, target_valid.T, target_test.T

def l2_distance(a, b):
  """Code taken from A2
  Computes the Euclidean distance matrix between a and b.
  """

  if a.shape[0] != b.shape[0]:
      raise ValueError("A and B should be of same dimensionality")

  aa = np.sum(a**2, axis=0)
  bb = np.sum(b**2, axis=0)
  ab = np.dot(a.T, b)

  return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)

def run_knn(k, train_data, train_labels, valid_data):
  """Code taken from A2
  Uses the supplied training inputs and labels to make
  predictions for validation data using the K-nearest neighbours
  algorithm.

  Note: N_TRAIN is the number of training examples,
        N_VALID is the number of validation examples, 
        and M is the number of features per example.

  Inputs:
      k:            The number of neighbours to use for classification 
                    of a validation example.
      train_data:   The N_TRAIN x M array of training
                    data.
      train_labels: The N_TRAIN x 1 vector of training labels
                    corresponding to the examples in train_data 
                    (must be binary).
      valid_data:   The N_VALID x M array of data to
                    predict classes for.

  Outputs:
      valid_labels: The N_VALID x 1 vector of predicted labels 
                    for the validation data.
  """

  dist = l2_distance(valid_data.T, train_data.T)
  nearest = np.argsort(dist, axis=1)[:,:k]

  train_labels = train_labels.reshape(-1)
  valid_labels = train_labels[nearest]

  # note this only works for binary labels
  valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
  valid_labels = valid_labels.reshape(-1,1)

  return valid_labels