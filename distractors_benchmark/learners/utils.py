import tensorflow as tf


def add_dummy_dimension(x):
    return x[..., None]


@tf.function
def cosine_similarity(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Computes cosine similarity between all pairs of vectors in x and y."""
    x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
    similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
    similarity_matrix /= (
        tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + 1e-9
    )
    return similarity_matrix


@tf.function
def pdist_l2(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """Computes pairwise euclidian distances between the rows of two tensors.

    Args:
        A (tf.Tensor): An n by k tensor.
        B (tf.Tensor): An m by k tensor.

    Returns:
        tf.Tensor: An n by m tensor.
    """
    assert A.shape.as_list() == B.shape.as_list()

    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    return tf.sqrt(
        row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B + 1e-3
    )  # TODO: angelos review 1e-3 very large?
