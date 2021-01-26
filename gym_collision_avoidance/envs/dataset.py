import numpy as np
from gym_collision_avoidance.envs.utils import DataHandlerLSTM

class Dataset():
  ## Load Dataset
  # Create Datahandler class
  data_prep = DataHandlerLSTM.DataHandlerLSTM('zara_01')
  # Only used to create a map from png
  # Make sure this parameters are correct otherwise it will fail training and ploting the results
  map_args = {"file_name": 'map.png',
              "resolution": 0.1,
              "map_center": np.array([8., 8.]),
              "map_size": np.array([16., 16.]), }
  # Load dataset
  data_prep.processData(**map_args)