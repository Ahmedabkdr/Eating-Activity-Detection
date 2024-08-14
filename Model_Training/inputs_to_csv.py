import pandas as pd

from accel_to_observations import accel_to_observations


# assumes there are as many is_eating files as there are accel files
def inputs_to_csv(accel_files, is_eating_files, inputs_file_name):
    inputs = pd.DataFrame()
    # adding the observations where the person is eating
    for index, file in enumerate(accel_files):
        inputs = pd.concat([inputs, accel_to_observations(file, is_eating_files[index], "Eating Files")], axis=0)
    inputs_as_array = inputs.to_numpy()
    inputs.to_csv(inputs_file_name)
