""" helper functions for this project """
from random import shuffle
from pandas import DataFrame
import numpy as np
import pandas as pd
import logging as lg
from scipy.spatial.distance import euclidean
import re


def check_type(obj, class_or_tuple, var_name):
    """
    check type and raise error if fails 
    see isinstance for more information

    Parameters
    ----------
    obj : object
        object under test
    class_or_tuple : class or tuple (A, B, ...)
        for tuple => check_type(x, A) or check_type(x, B) ...
    var_name : str
        variable name for which this error occurred
    """

    if not isinstance(obj, class_or_tuple):
        print('invalid data', obj)
        raise TypeError('see docs for ' + var_name)


def shuffle_frame(frame):
    """
    randomize rows for each location.

    Parameters
    ----------
    frame : Pandas Data frame
            data frame will be shuffled
    """

    # make two empty copies of raw data frame
    # one for test and the other for train
    temp = DataFrame(columns=frame.columns)

    # xy coordinates
    rps = frame[['x', 'y']].drop_duplicates().values

    # for each location
    for [x, y] in rps:
        # get the indices of rows that represent this location
        indices = list(frame.loc[(frame['x'] == x) & (frame['y'] == y)].index)

        shuffle(indices)

        # append rows into train and test
        temp = temp.append(frame.loc[indices],
                           ignore_index=True,
                           verify_integrity=True)

    # check final shapes
    # number of columns should be same in all three
    assert temp.shape[1] == frame.shape[1], "shuffle column mismatch"
    assert temp.shape[0] == frame.shape[0], "shuffle row mismatch"

    # return the training and testing sets as data frames
    return temp


def split_frame(frame, split=0.8, shuffle=True, target="label", seed=42):
    """
    split into two dataframes one for training and other for testing
    describes train-test partition of data
    Parameters
    ----------
    frame : Pandas Data frame
    dataframe will be split

    split : float between 0-1
    """

    # make two empty copies of raw data frame
    # one for test and the other for train
    train = pd.DataFrame(columns=frame.columns)
    test = pd.DataFrame(columns=frame.columns)

    # unique reference points (rps)
    rps = frame[[target]].drop_duplicates().values.flatten()

    # for each location
    for lbl in rps:
        # get the indices of rows that represent this location
        indices = list(frame.loc[frame[target] == lbl].index)

        if shuffle:
            np.random.shuffle(indices, )

        # get the indices of raw data rows for train and test
        split_pos = int(len(indices) * split)

        # use the split positions above to identify splits
        train_indices = indices[0: split_pos]
        test_indices = indices[split_pos:]

        # append rows into train and test
        train = train.append(frame.loc[train_indices],
                             ignore_index=True,
                             verify_integrity=True)

        test = test.append(frame.loc[test_indices],
                           ignore_index=True,
                           verify_integrity=True)

    # check final shapes
    # number of columns should be same in all three
    assert train.shape[1] == test.shape[1] == frame.shape[1], \
        "error splitting raw data"

    # return the training and testing sets as dataframes
    return train, test


def frame_ref_pt_index(frame, indx):
    indexes = refpoints_as_df(frame).reset_index(drop=True)

    return at_refpoint(frame, list(indexes.loc[indx].values))


def frame_at_xy(frame, x, y):
    return at_refpoint(frame, [x, y])


def at_refpoint(data_frame: DataFrame, xy_point: list) -> DataFrame:
    """ return rows at xy location """
    return data_frame.loc[
        (data_frame['x'] == xy_point[0]) &
        (data_frame['y'] == xy_point[1])
    ]


def refpoints_as_df(frame):
    """
    get all unique ref points in the frame
    :param frame:
    :return:
    """
    return frame.loc[:, ['x', 'y']].drop_duplicates()


def split_df_by_rps(df):
    rps = refpoints_as_df(df)

    splits = []

    for [x, y] in rps.values:
        splits.append((frame_at_xy(df, x, y)))

    return splits


def set_samples_per_ref_pt(frame, num_samples):
    """  
    Get a frame with given number of samples per reference point

    frame: dataframe
        should have columns x and y
    num_samples: int
        number of samples per ref to keep
    """
    import pandas as pd

    # will store new frame here
    new_f = pd.DataFrame(columns=frame.columns)

    # iterate over each location
    for (i, xy_point) in frame[['x', 'y']].drop_duplicates().iterrows():

        # frame at this location
        at_ref_df = frame.loc[(frame['x'] == xy_point[0]) &
                              (frame['y'] == xy_point[1])]

        # get n rows from dataframe at given location
        sample_df = at_ref_df.sample(n=num_samples)

        # store in new frame
        new_f = new_f.append(sample_df)

    # return frame
    return new_f


def make_images(vectors, fmt='tf', force_shape=None):
    """convert into N, H, W, C, with C=1 or BW img
    assumes H == W by default, 
    :param vectors: 2d array
    :param fmt: 'tf' (N, H, W, 1) or 'tf' (N, 1, H, W)
    :param force_shape: force shape (H, W), C is always 1
    :returns: returns images in 'th' or 'tf' format
    :rtype: numpy array
    """
    closest_sqaure = np.square(np.ceil(np.sqrt(vectors.shape[1])))
    
    if force_shape is None:
        req_pad = int(closest_sqaure - vectors.shape[1])
        img_size = int(np.sqrt(closest_sqaure))
        h, w = img_size, img_size
    elif force_shape is not None and len(force_shape) == 2:
        h, w = force_shape
        req_pad = int((h * w) - vectors.shape[1])
    else:
        lg.warning("invalid input for force_shape. auto computing...")
        req_pad = int(closest_sqaure - vectors.shape[1])
        img_size = int(np.sqrt(closest_sqaure))
        h, w = img_size, img_size

    lg.debug(f"make_images()")
    lg.debug(f"before padding shape: {vectors.shape}")
    lg.debug(f"closest square: {closest_sqaure}")
    lg.debug(f"required padding: {req_pad}")
    lg.debug(f"img size: {h, w}")

    # pad if required
    if req_pad != 0:
        # do padding here
        vectors = np.hstack((
            vectors,
            np.zeros((vectors.shape[0], req_pad))
        ))

    if fmt == 'tf':
        return vectors.reshape((-1, h, w, 1))
    elif fmt == 'th':
        return vectors.reshape((-1, 1, h, w))
    else:
        lg.error("Invalid fmt argument")
        raise Exception


def compute_distances(u, v, lbl2cords=None):
    """
    wrapper on eucledian to compute distance over two sets of arrays
    if lbl2crds then assumes u and v are flatt array of labels
    """

    if lbl2cords is not None:
        new_u = []
        for lbl in np.array(u).flatten():
            new_u.append(lbl2cords[lbl])
        new_v = []
        for lbl in np.array(v).flatten():
            new_v.append(lbl2cords[lbl])

        # overwrite things
        u = np.array(new_u)
        v = np.array(new_v)

    dists = []
    for x, y in zip(u, v):
        dists.append(euclidean(x, y))

    return np.array(dists)



def get_visible_waps(records: pd.DataFrame, missing_val=-100, wap_re=r"WAP_\d+"):
    """
    list of visible waps
    records: for a given df; any number of rows
    """
    # extract col names
    wap_cols = []
    for col in records.columns:
        if re.match(wap_re, col):
            wap_cols.append(col)
    records = records[wap_cols]

    return records.replace(to_replace=missing_val,
                            value=np.nan).dropna(axis=1,
                                                how='all').columns


def label2coords_builder(arr: np.array, scale=1):
    """
    Array such that first column is label, and others are coords
    """

    lbl2coords = {}

    for row in arr:
        lbl2coords[int(row[0])] = np.array([*row[1:],])/scale

    return lbl2coords



if __name__ == "__main__":
    print("Running test in helpers.")

    x = np.random.random(9).reshape((1, 9))
    print(x, x.shape)

    print("unforced make image")
    ux = make_images(x)
    print(ux, ux.shape)


    print("forced make image")
    ux = make_images(x, force_shape=(4, 4))
    print(ux, ux.shape)

    print("forced make image")
    ux = make_images(x, force_shape=(4, 3))
    print(ux, ux.shape)