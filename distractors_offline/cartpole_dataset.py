import numpy as np
import reverb
import tensorflow as tf


def _make_reverb_sample(
    o_t: tf.Tensor,
    a_t: tf.Tensor,
    r_t: tf.Tensor,
    d_t: tf.Tensor,
    o_tp1: tf.Tensor,
) -> reverb.ReplaySample:
    """Create Reverb sample with offline data.

    Args:
      o_t: Observation at time t.
      a_t: Action at time t.
      r_t: Reward at time t.
      d_t: Discount at time t.
      o_tp1: Observation at time t+1.
      a_tp1: Action at time t+1.
      extras: Dictionary with extra features.

    Returns:
      Replay sample with fake info: key=0, probability=1, table_size=0.
    """
    info = reverb.SampleInfo(
        key=tf.constant(0, tf.uint64),
        probability=tf.constant(1.0, tf.float64),
        table_size=tf.constant(0, tf.int64),
        priority=tf.constant(1.0, tf.float64),
    )
    data = (o_t, a_t, r_t, d_t, o_tp1)
    return reverb.ReplaySample(info=info, data=data)


def _tf_example_to_reverb_sample(tf_example: tf.train.Example) -> reverb.ReplaySample:
    """Create a Reverb replay sample from a TF example."""
    # Process data.
    # Create a description of the features.

    feature_description = {
        "r_t": tf.io.FixedLenFeature([], tf.float32, default_value=0),
        "o_tm1": tf.io.FixedLenFeature([6], tf.float32),
        "a_tm1": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "o_t": tf.io.FixedLenFeature([6], tf.float32),
        "d_t": tf.io.FixedLenFeature([], tf.float32, default_value=0),
        "step": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "episode": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    data = tf.io.parse_single_example(tf_example, feature_description)

    return (
        data["o_tm1"],
        data["a_tm1"],
        data["r_t"],
        data["d_t"],
        data["o_t"],
    )


def dataset(add_distractor, n_distractors) -> tf.data.Dataset:
    dataset_path = "cartpole_with_noise_5_step.npy"
    np_data = np.load(dataset_path, allow_pickle=True)
    print(f"Loaded dataset with shape {str(np_data.shape)} from {dataset_path}")

    def data_generator():
        for row in np_data:
            yield tuple(row)

    file_ds = (
        tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(6), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(6), dtype=tf.float32),
            ),
        )
        .map(_make_reverb_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(256, drop_remainder=True)
    )

    if add_distractor and n_distractors:
        file_ds = file_ds.map(
            add_distractor,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    file_ds = file_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return file_ds.cache().repeat()


class VectorizedDistractors:
    """Adds `n_distractors` distractions of `distractors_type` to each transition being loaded into memory."""

    def __init__(self, n_distractors, distractors_type):
        self._n_distractors = n_distractors
        self._distractors_type = distractors_type
        self.timescale = 0.01
        print(f"Initializing {n_distractors} {distractors_type} distractors")

        # Initialize distractor states/parameters
        assert self._distractors_type in (
            "gaussian",
            "sine",
            "action-walk",
        ), f"You have {self._n_distractors} distractors but `_distractors_type`={distractors_type} is not a valid option."

        if self._distractors_type == "action-walk":
            self._distractor_step = tf.random.uniform(
                (self._n_distractors,), minval=-0.1, maxval=0.1
            )

    def distract_observation(self, o_t):
        """Adds distractors to an observation received in online evaluation"""
        if self._n_distractors == 0:
            return o_t

        if self._distractors_type == "gaussian":
            distractors_t = tf.random.normal([1, self._n_distractors])
        elif self._distractors_type == "sine":
            random_phases = tf.random.uniform(
                (
                    1,
                    self._n_distractors,
                ),
                minval=0,
                maxval=2 * np.pi,
            )
            distractors_t = tf.math.sin(random_phases)
        elif self._distractors_type == "action-walk":
            # Distractor `i` is a action-based walk: distractor[i] += step * action (action-1 = -1,0,or 1)
            distractors_t = tf.random.uniform(
                (
                    1,
                    self._n_distractors,
                ),
                minval=-2,
                maxval=2,
            )
        else:
            raise Exception()

        return tf.concat(
            [o_t, tf.convert_to_tensor(distractors_t, dtype=tf.float32)],
            axis=-1,
        )

    def add_distractors(self, sample):
        """Adds distractors to a dataset sample"""
        if self._n_distractors > 0:
            (o_tm1, a_t, r_t, d_t, o_t) = sample.data
            batch_size = tf.shape(o_tm1)[0]

            if self._distractors_type == "gaussian":
                distractors_tm1 = tf.random.normal([batch_size, self._n_distractors])
                distractors_t = tf.random.normal([batch_size, self._n_distractors])
            elif self._distractors_type == "sine":
                random_phases = tf.random.uniform(
                    (
                        batch_size,
                        self._n_distractors,
                    ),
                    minval=0,
                    maxval=2 * np.pi,
                )
                distractors_tm1 = tf.math.sin(random_phases)
                distractors_t = tf.math.sin(random_phases + 2 * self.timescale)
            elif self._distractors_type == "action-walk":
                # Distractor `i` is a action-based walk: distractor[i] += step * action (action-1 = -1,0,or 1)
                distractor_starting = tf.random.uniform(
                    (
                        batch_size,
                        self._n_distractors,
                    ),
                    minval=-2,
                    maxval=2,
                )
                distractors_tm1 = distractor_starting
                distractors_t = (
                    distractor_starting
                    + tf.cast(tf.expand_dims(a_t, -1) - 1, dtype=tf.float32)
                    * self._distractor_step
                )
            else:
                raise Exception()

            o_t = tf.concat(
                [o_t, tf.convert_to_tensor(distractors_t, dtype=tf.float32)],
                axis=-1,
            )
            o_tm1 = tf.concat(
                [o_tm1, tf.convert_to_tensor(distractors_tm1, dtype=tf.float32)],
                axis=-1,
            )
            # Do not include episode id
            return sample._replace(data=(o_tm1, a_t, r_t, d_t, o_t))
        else:
            return sample


def subsample_and_shuffle_records():
    filenames = [
        f"/vol/bitbucket/ac7117/cartpole_data/five-step-transitions/cartpole_run_{r}.tfrecord"
        for r in range(1, 5)
    ]
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(
        _tf_example_to_reverb_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    data_iterator = ds.as_numpy_iterator()

    data = list()

    # subsampling_ratio = 1
    for n, datum in enumerate(data_iterator):
        if n % 100000 == 0:
            print(n)
        # if n % subsampling_ratio == 0:
        data.append(datum)
    # shuffle
    np.random.shuffle(data)
    np.save("cartpole_with_noise_5_step.npy", data)


if __name__ == "__main__":
    subsample_and_shuffle_records()
    # n = 2
    # distractors = VectorizedDistractors(n, 'sine')
    # dset = iter(dataset(distractors.add_distractors))
    # while True:
    #     batch = next(dset)
    #     o_tm1 = batch.data[0][:10, -n:]
    #     o_t = batch.data[4][:10, -n:]
    #     action  =batch.data[1]


# class Distractors:
#     """Adds `n_distractors` distractions of `distractors_type` to each transition being loaded into memory."""

#     def __init__(self, n_distractors, distractors_type, vectorized):
#         self._vectorized = vectorized
#         self._n_distractors = n_distractors
#         self._distractors_type = distractors_type
#         self.time_elapsed = tf.Variable(0.0)
#         self.timescale = 0.01
#         self.init_phases = lambda n: tf.random.uniform((n,), minval=0, maxval=2 * np.pi)
#         print(f"Initializing {n_distractors} {distractors_type} distractors")

#         # Initialize distractor states/parameters
#         assert self._distractors_type in (
#             "gaussian",
#             "sine",
#             "action-walk",
#         ), f"You have {self._n_distractors} distractors but `_distractors_type`={distractors_type} is not a valid option."

#         if self._distractors_type == "sine":
#             self._distractor_phases = tf.Variable(
#                 self.init_phases(self._n_distractors), dtype=tf.float32
#             )
#         elif self._distractors_type == "action-walk":
#             self._distractor_starting, self._distractor_step = init_walk_params(
#                 self._n_distractors
#             )
#             self._distractor_starting = tf.Variable(
#                 self._distractor_starting, dtype=tf.float32
#             )
#             self._distractor_step = tf.Variable(self._distractor_step, dtype=tf.float32)
#             self._distractor_state = tf.Variable(self._distractor_starting)

#     def add_distractors(self, sample):
#         if self._n_distractors > 0:
#             (o_tm1, a_t, r_t, d_t, o_t) = sample.data
#             # Reset:
#             if d_t == 0:
#                 if self._distractors_type == "sine":
#                     self._distractor_phases.assign(
#                         self.init_phases(self._n_distractors)
#                     )
#                 if self._distractors_type == "action-walk":
#                     self._distractor_state.assign(self._distractor_starting)

#             if self._distractors_type == "gaussian":
#                 # Vectorized implementation
#                 batch_size = tf.shape(o_tm1)[0]
#                 distractors_tm1 = tf.random.normal([batch_size, self._n_distractors])
#                 distractors_t = tf.random.normal([batch_size, self._n_distractors])
#             elif self._distractors_type == "sine":
#                 # Each distractor is a discretized sinusoid, with a phase that is initialized on .reset()
#                 # TODO: variable frequency
#                 distractors_tm1 = tf.math.sin(
#                     2 * self.time_elapsed + self._distractor_phases
#                 )
#                 self.time_elapsed.assign_add(self.timescale)
#                 distractors_t = tf.math.sin(
#                     2.0 * self.time_elapsed + self._distractor_phases
#                 )
#             elif self._distractors_type == "action-walk":
#                 # Distractor `i` is a action-based walk: distractor[i] += step * action (action-1 = -1,0,or 1)
#                 distractors_tm1 = self._distractor_state
#                 self.time_elapsed.assign_add(self.timescale)
#                 self._distractor_state.assign_add(
#                     tf.cast(a_t - 1, dtype=tf.float32) * self._distractor_step
#                 )
#                 self._distractor_state.assign(
#                     tf.math.floormod(self._distractor_state + 2.0, 4.0) - 2.0
#                 )
#                 distractors_t = self._distractor_state
#             else:
#                 raise Exception()

#             concat_axis = -1 if self._vectorized else 0
#             o_t = tf.concat(
#                 [o_t, tf.convert_to_tensor(distractors_t, dtype=tf.float32)],
#                 axis=concat_axis,
#             )
#             o_tm1 = tf.concat(
#                 [o_tm1, tf.convert_to_tensor(distractors_tm1, dtype=tf.float32)],
#                 axis=concat_axis,
#             )
#             # Do not include episode id
#             return sample._replace(data=(o_tm1, a_t, r_t, d_t, o_t))
#         else:
#             return sample


# def dataset(
#     path: str,
#     run: str,
#     n_distractors: int,
#     distractors_type: str,
#     shuffle_buffer_size: int = 1000000,
#     filter_episodes: int = 0,
#     raw_analysis: bool = False,
# ) -> tf.data.Dataset:

#     assert "-" in run
#     from_run, to_run = run.split("-")
#     from_run, to_run = int(from_run), int(to_run)
#     print(f"Using runs {str(from_run)} to {str(to_run)}.")

#     filenames = [
#         f"{path}cartpole_run_{r}.tfrecord" for r in range(from_run, to_run + 1)
#     ]

#     print("`TFRecordDataset` - starting", filenames)
#     file_ds = tf.data.TFRecordDataset(filenames)
#     print("`TFRecordDataset` - done")

#     file_ds = file_ds.map(
#         _tf_example_to_reverb_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
#     )
#     # Filter by episode number within run
#     if filter_episodes:
#         print(f"Skipping any episode_id < {filter_episodes}")
#         file_ds = file_ds.filter(lambda x: x.data[-1] > filter_episodes)

#     file_ds = file_ds.map(lambda x: x._replace(data=x.data[:5]))

#     if raw_analysis:
#         return file_ds

#     if n_distractors > 0:
#         vectorized = distractors_type == "gaussian"
#         add_distractors = Distractors(n_distractors, distractors_type, vectorized)

#         if vectorized:
#             file_ds = file_ds.shuffle(shuffle_buffer_size, seed=42)
#             file_ds = file_ds.batch(256)

#         file_ds = file_ds.map(
#             add_distractors.add_distractors,
#             num_parallel_calls=tf.data.experimental.AUTOTUNE,
#         )

#         if not vectorized:
#             file_ds = file_ds.shuffle(shuffle_buffer_size, seed=42)
#             file_ds = file_ds.batch(256)
#     else:
#         file_ds = file_ds.shuffle(shuffle_buffer_size, seed=42)
#         file_ds = file_ds.batch(256)

#     file_ds = file_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     return file_ds.repeat()
