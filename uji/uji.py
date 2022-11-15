# module to access UJI db as pandas df
import os
import glob
import re
import pandas as pd
import numpy as np
import logging as lg
from tqdm import tqdm
import warnings
from scipy.spatial.distance import euclidean
import json
import pickle


class UJI():
    """
    UJI class for accessing data

    IMPORTANT: MAKE SURE CACHE_DIR and DATA_DIR ARE CORRECT

    NOTE: It is recommended to build the cache for this class using
    UJI.build_cache_complete()
    Then, instead of using constructor, use:
    UJI.from_cache(mode, month)
    This is FASTER!

    properties:
    -----------
    mode: 'trn' or 'tst' 
    month: integer month 
    collection_instances: tuple of instances in month
    records: dataframe
    COL_NAMES: column names other than WAPs

    functions:
    ----------
    extract_location_label: extract location label from ID
    extract_id_info: extract info from ID as dict
    filter_record: filter rows by column values
    filter_record_range: get rows have value between range
    """
    CACHE_DIR = "uji/db_cache"

    def __init__(self, mode, month,
                 DATA_DIR="uji/db",
                 verbose=False):
        """
        args:
        -----
        mode (str): "train" or "test"
        month (int): int month 1-15
        DATA_DIR (str): full path to directory where "db" is located
        """

        # number of WAPS expected in DB
        self.__NUM_WAPS = 620
        self.__MISSING_VAL = 100
        self.__TO_MISSING_VAL = -100
        self.__DATA_DIR = DATA_DIR
        self.__VERBOSE = verbose
        self.__CUSTOM_LABEL_FILE = DATA_DIR + "/../location_labels.dict"

        ###########################
        # Cache objects
        ##########################

        ## check inputs ##
        # mode
        if mode == "train":
            self.mode = "trn"
        elif mode == "test":
            self.mode = "tst"
        else:
            exit(f"""{str(self.__class__)}: invalid mode entered.
            valid choices: train or test""")

        # month
        if type(month) != int or month < 1 or month > 15:
            exit(f"""{str(self.__class__)}: invalid month entered.
            valid choices: 1-15""")
        else:
            self.month = str(month).zfill(2)
        ###################

        # get train data instances
        self.collection_instances = self.__get_collection_instances(self.month,
                                                                    self.mode,
                                                                    base_path=self.__DATA_DIR)

        # build the records dataframe for this class
        self.__records = self.__build_records()

        # make column names properties
        # this will make every annotated column appear as class property
        for col in self.annot_columns:
            setattr(self, col, col)

        # use custom labels; assumes label column was already built
        # hopefully under build records
        self.uji_lbl2coord, self.uji_coord2lbl = self.__build_location_dicts(
            "UJI_LABEL")

        # use custom labels; assumes label column was already built
        # hopefully under build records
        self.lbl2coord, self.coord2lbl = self.__build_location_dicts("LABEL")

        lg.debug(self.__records)

    @classmethod
    def from_cache(cls, mode, month, cache_dir=None):
        """Class method that can be used as a contructor
        NOTE: cls object is never really used
        :param cls: classmethod
        :param mode: "train" or "test"
        :param month: 1 - 15 integer
        :returns: UJI object
        :rtype: UJI

        """

        if cache_dir is None:
            cache_dir = UJI.CACHE_DIR

        file_name = cache_dir + f"/{mode}_{month}.pkl"
        return pickle.load(open(file_name, "rb"))

    def __build_records(self):
        """
        build the dataframe associated with this mode and month
        """
        # for each collection instance;
        # fetch the files and create on large dataframe
        month_data_np = None

        if lg.root.level == lg.DEBUG:
            lg.debug("Reading csv files")
            instances = tqdm(self.collection_instances)
        else:
            instances = self.collection_instances

        for ci in instances:

            # create the base file path
            path_base = os.path.join(self.__DATA_DIR, self.month)

            # create paths to all files
            path_crd = os.path.join(
                path_base, f"{self.mode}{str(ci).zfill(2)}crd.csv")
            path_ids = os.path.join(
                path_base, f"{self.mode}{str(ci).zfill(2)}ids.csv")
            path_rss = os.path.join(
                path_base, f"{self.mode}{str(ci).zfill(2)}rss.csv")
            path_tms = os.path.join(
                path_base, f"{self.mode}{str(ci).zfill(2)}tms.csv")

            # read each file as a numpy array
            crd_np = np.genfromtxt(
                path_crd, delimiter=",", dtype=str).reshape((-1, 3))

            ids_np = np.genfromtxt(
                path_ids, delimiter=",", dtype=str).reshape((-1, 1))

            rss_np = np.genfromtxt(path_rss, delimiter=",", dtype=str)
            rss_np = rss_np.reshape((rss_np.shape[0], self.__NUM_WAPS))

            tms_np = np.genfromtxt(
                path_tms, delimiter=",", dtype=str).reshape((-1, 1))

            # append a column representing collection instance
            collection_instance_np = np.zeros(
                (crd_np.shape[0], 1), dtype="int") + ci
            collection_instance_np = collection_instance_np.astype(str)

            # append each numpy array
            ci_data_np = np.hstack((
                rss_np,
                crd_np,
                ids_np,
                tms_np,
                collection_instance_np
            ))

            ## clear unused variables ##
            del rss_np,
            del crd_np,
            del ids_np,
            del tms_np,
            del collection_instance_np

            # if data is empty, then populate
            # else append below data
            if month_data_np is None:
                month_data_np = np.copy(ci_data_np)
            else:
                month_data_np = np.vstack((month_data_np, ci_data_np))

        # print some info about records about records
        if self.__VERBOSE:
            print(f"""Record Detils:
            Mode: {self.mode}
            Month: {self.month}
            Collection Sessions: {self.collection_instances}
            Record Shape: {month_data_np.shape}
            """)

        # build column names
        wap_col_names = {}
        for i in range(self.__NUM_WAPS):
            wap_col_names["WAP_" + str(i)] = "int"

        other_cols = {
            "CORD_X": "float",
            "CORD_Y": "float",
            "FLOOR": "int",
            "ID": "string",
            "TIMESTAMP": "int",
            "COLLECTION_INSTANCE": "int"
        }

        # keep a list of columns
        self.__wap_cols = tuple(wap_col_names.keys())
        self.__annot_cols = tuple(other_cols.keys())

        # create dataframe
        records = pd.DataFrame(month_data_np,
                               columns=list(wap_col_names.keys()) +
                               list(other_cols.keys()),
                               )

        # apply formats
        records = records.astype(dtype={**wap_col_names, **other_cols})

        # set apt format for datetime
        records["TIMESTAMP"] = pd.to_datetime(records["TIMESTAMP"],
                                              format="%Y%m%d%H%M%S%f")

        ## delete orginal np array to clear space ##
        del month_data_np

        # create new column for locations
        records["UJI_LABEL"] = records.apply(lambda row:
                                             UJI.extract_location_label(
                                                 row["ID"]),
                                             axis=1)

        # create new column for custom labels
        records["LABEL"] = self.__apply_custom_labels(records)
        records["LABEL"] = records["LABEL"].astype(np.int)

        # create a new column called DATE
        records[["DATE"]] = records.TIMESTAMP.dt.date

        # repalce missing values with something more workable
        records[self.all_waps] = records[self.all_waps].replace(
            to_replace=self.__MISSING_VAL,
            value=self.__TO_MISSING_VAL,
        )

        return records

    def __build_location_dicts(self, label_column):
        """
        create two dicts;
        location label to coord;
        coord to label;
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lbl2coord = self.__records[[label_column,   # contains uji location label
                                        "FLOOR",        # current floor
                                        "CORD_X",       # x coord on this floor
                                        "CORD_Y"]]      # y coord on this floor

            # set indec to LOC before converting to dict
            lbl2coord = lbl2coord.drop_duplicates().set_index(label_column).T.to_dict("list")

        # invert to get the other dict
        coord2lbl = {}
        for key, value in lbl2coord.items():
            coord2lbl[str(value)] = key

        return lbl2coord, coord2lbl

    def __get_file_paths(self, path_pattern: str):
        """
        get names of files matching a pattern
        examples: "db/**/trn*crd.csv", f"db/{month}/trn*crd.csv"
        """
        for name in glob.glob(path_pattern, recursive=True):
            yield name

    def __get_collection_instances(self, month: int, mode: str, base_path=""):
        """
        get the instance from a file name
        month (int): month in integer 
        mode (str): trn or tst
        """
        instances = []
        # TODO: use os.join; so it works in windows
        for crd_path in self.__get_file_paths(f"{base_path}/{str(month).zfill(2)}/{mode}*crd.csv"):
            (crd_month, crd_inst) = re.match(f"{base_path}/(\d+)/{mode}(\d+)crd.csv",
                                             crd_path).groups()
            if crd_month != self.month:
                exit(f"""{str(self.__class__)}: invalid month found.
                {crd_path} does not match {self.month}""")

            instances.append(int(crd_inst))

        return tuple(sorted(instances))

    #### PROPERTY FUNCTIONS ####
    @staticmethod
    def uji_floor_mat_and_dict():
        floor_mat = np.full((17, 11), np.nan)
        label2ij = {}

        # fill column 3 and 7 - depicting central paths
        floor_mat[:, 3] = np.arange(0, 17)
        floor_mat[:, 7] = np.arange(33, 16, -1)

        # build label2ij for above cols
        for i in range(0, 17):
            label2ij[i] = (i, 3)

        k = 0
        for i in range(33, 16, -1):
            label2ij[i] = (k, 7)
            k += 1

        # fill three sections of columns
        start = 34
        for k in [0, 4, 8]:
            for i in range(1, 17, 2):
                for j in range(k, k + 3):
                    floor_mat[i, j] = start

                    # dict for quick search
                    label2ij[start] = (i, j)

                    start += 1

        return floor_mat, label2ij

    @property
    def full_walk(self):
        """list describing a walk over every label
        refer to test_uji.png 
        :returns: list
        :rtype: 
        """
        vert_walk_1 = list(range(1, 18))
        vert_walk_2 = list(range(18, 35))

        left = np.array(range(35, 59)).reshape((-1, 3))
        mid = np.array(range(59, 83)).reshape((-1, 3))
        right = np.array(range(83, 107)).reshape((-1, 3))

        horizontal_walk = []
        uwl = 2
        uwr = 33
        for lr, mr, rr in zip(left, mid, right):

            lr = list(lr)
            mr = list(mr)
            rr = list(rr)

            hw = lr + [uwl] + mr + [uwr] + rr

            horizontal_walk += hw

            uwl += 2
            uwr -= 2

        # adjust for new labels
        w = horizontal_walk + list(reversed(horizontal_walk)) + \
            vert_walk_1 + list(reversed(vert_walk_1)) + \
            vert_walk_2 + list(reversed(vert_walk_2))

        w = np.array(w) - 1

        return list(w)

    @property
    def label_edges(self):
        """returns array describing closeby nodes
        :returns: 
        :rtype: shape (-1, 2) numpy array 
        """
        edges = np.array([[0, 1]]).astype(np.int)

        ranges = [
            [(1, 16, 1), (2, 17, 1)],
            [(17, 33, 1), (18, 34, 1)],
            [(34, 56, 3), (35, 57, 3)],
            [(35, 57, 3), (36, 58, 3)],
            [(58, 80, 3), (59, 81, 3)],
            [(59, 81, 3), (60, 82, 3)],
            [(82, 104, 3), (83, 105, 3)],
            [(83, 105, 3), (84, 106, 3)],
            [(36, 58, 3), (1, 16, 2)],
            [(1, 16, 2), (58, 80, 3)],
            [(60, 82, 3), (32, 17, -2)],
            [(32, 17, -2), (82, 104, 3)]
        ]

        # build edges
        for r1, r2 in ranges:
            for i, j in zip(np.arange(*r1), np.arange(*r2)):
                e = np.array([[i, j]])
                edges = np.vstack((edges, e))

        return edges.astype(np.int)

    @property
    def records(self):
        """
        getter function for self.__records
        """
        return self.__records

    @property
    def all_waps(self):
        """
        getter for list of all wap columns
        """
        return list(self.__wap_cols)

    @property
    def annot_columns(self):
        """
        getter for all annotation columns
        these are all columns that are not waps
        """
        return list(self.__annot_cols)

    #### PUBLIC FUNCTIONS ####

    def intersect_visible_waps(self, **filters):
        """Get common WAPs seen in each Collection Instance of this month
        :param filters: arguments passed to filter records function
        :returns: 
        :rtype: 
        """
        set_intersect = None

        for ci in self.collection_instances:
            train_mx_c1 = self.filter_record(
                **filters,
                COLLECTION_INSTANCE=ci
            )

            visible_waps = self.get_visible_waps(records=train_mx_c1)

            if set_intersect is None:
                set_intersect = set(visible_waps)
            else:
                set_intersect = set_intersect.intersection(set(visible_waps))

        return list(set_intersect)

    def filter_record(self,
                      CORD_X=None,
                      CORD_Y=None,
                      FLOOR=None,
                      ID=None,
                      TIMESTAMP=None,
                      DATE=None,
                      COLLECTION_INSTANCE=None,
                      UJI_LABEL=None,
                      LABEL=None,
                      records=None,
                      ):
        """
        filter the record for this month based on some arguments
        kwargs:
        -------
        records (pd.Dataframe): use this dataframe instead of base dataframe in class

        returns:
        --------
        pd.Dataframe
        """

        if records is None:
            records = self.__records.copy()

        # check every column name
        for column_name in self.__annot_cols:

            # if column name is in locals variables
            if column_name in list(locals().keys()) and \
                    locals().get(column_name, None) is not None:

                # get value of arguemnt passed to this function
                col_value = locals()[column_name]

                # check if value passed is iterable; if not make it one
                if not isinstance(col_value, list):
                    col_value = [col_value]

                # extract all records where columns have these values; iteratively
                records = records.loc[records[column_name].isin(col_value)]

        # finally, return the records
        return records

    def filter_record_range(self, **kwargs):
        """
        get records within this range
        expected input
        'COLUMN_NAME' = (start, end)
        """
        records = kwargs.get('records', self.__records.copy())

        for column_name, (start_value, end_value) in kwargs.items():
            records = records.loc[(records[column_name] >= start_value) &
                                  (records[column_name] <= end_value)]

        return records

    def get_coord_converters(self, floor, facing=None):
        """
        getter for two dicts
        """
        # make dict for this floor
        lbl2coord = {}
        for lbl, [fl, x, y] in self.uji_lbl2coord.items():

            if fl == floor:
                lbl2coord[lbl] = [x, y]

        all_labels = list(lbl2coord.keys())

        # make a sublist of labels that you may want
        if facing is True:
            # get first half
            labels = all_labels[:int(len(all_labels)/2) + 1]
        elif facing is False:
            # get second half
            labels = all_labels[int(len(all_labels)/2):]
        else:
            labels = all_labels.copy()

        # create a sub-dict contains labels based on facing
        lbl2coord_facing = {k: lbl2coord[k] for k in lbl2coord if k in labels}

        coord2lbl = {}
        for lbl, value in lbl2coord_facing.items():
            coord2lbl[str(value)] = lbl

        return lbl2coord_facing, coord2lbl

    def get_visible_waps(self, records=None):
        """
        for the current collection month and instance
        list of visible waps
        records: for a given df; any number of rows
        """
        if records is None:
            records = self.__records.copy()

        # extract col names
        wap_cols = []
        for col in records.columns:
            if re.match(r"WAP_\d+", col):
                wap_cols.append(col)
        records = records[wap_cols]

        return records.replace(to_replace=self.__TO_MISSING_VAL,
                               value=np.nan).dropna(axis=1,
                                                    how='all').columns

    def __apply_custom_labels(self,
                              records,
                              custom_labels=None):
        """
        apply custom labels
        custom_labels (dict(dict)): dict of dict
        """

        if isinstance(custom_labels, str):
            custom_labels = json.load(open(custom_labels, "r"))
        elif isinstance(custom_labels, dict):
            pass
        elif custom_labels is None:
            custom_labels = json.load(open(self.__CUSTOM_LABEL_FILE, "r"))
        else:
            exit("invalid custom labels")

        def get_label(vec):
            x = str(vec[0])
            y = str(vec[1])
            return custom_labels[x][y]

        # create new function with column applied
        return pd.DataFrame(records[["CORD_X", "CORD_Y"]]).apply(get_label, axis=1)

    def labels_to_coords(self, labels: list, lbl2coord=None):
        """
        Convert labels to coords based on some dictionary of label to coord
        """
        if lbl2coord is None:
            lbl2coord = self.lbl2coord

        coords = []
        for lbl in np.array(labels).flatten():
            try:
                coords.append(lbl2coord[lbl])
            except:
                print(sorted(lbl2coord.keys()))
                raise
        return np.array(coords)

    ### STATIC METHODS FOR GENERAL UTILITY ###

    @staticmethod
    def build_cache(mode, month, cache_dir=None):
        """create the pkl file for UJI(mode, month) in cache

        :param mode: "train" or "test"
        :param month: 1 - 15
        """
        uji = UJI(mode, month)

        if cache_dir is None:
            cache_dir = UJI.CACHE_DIR

        file_name = cache_dir + f"/{mode}_{month}.pkl"
        pickle.dump(uji, open(file_name, "wb"))

    @staticmethod
    def build_cache_complete():
        """build the complete cache for this class
           For train and test modes and for each month within
        """
        for mode in tqdm(["train", "test"]):
            for month in tqdm(range(1, 16)):
                UJI.build_cache(mode, month)

    @staticmethod
    def compute_distances(u, v):
        """
        wrapper on eucledian to compute distance over two sets of arrays
        """

        dists = []
        for x, y in zip(u, v):
            dists.append(euclidean(x, y))

        return np.array(dists)

    @staticmethod
    def extract_location_label(sample_id: str):
        """
        extract the location label from sample id string
        see extract_id_info for more details
        """
        return UJI.extract_id_info(sample_id)["label"]

    @staticmethod
    def extract_id_info(sample_id: str):
        """
        get the label from id
        sample_id (str): String format for ID as in files like trn01ids.csv        
        example:
        --------
        sample id: 1 2 0 4 2 1 0 2 0 3

        0 1   2 3   4   5 6 7   8 9  # STRING INDICES
        1 2 | 0 4 | 2 | 1 0 2 | 0 3  # SAMPLE ID
        ---   ---   -   -----   ---
         |     |    |     |      |
         |     |    |     |      ---> Point’s Sample (1-6)
         |     |    |     ----------> Dataset’s Point (The amount depends on the dataset)
         |     |    ----------------> Dataset Type (1 for Training, 2 for Test)
         |     ---------------------> Month’s Dataset (The amount depends on month and type)
         ---------------------------> Month (1-15)  
        """

        return {
            "month": int(sample_id[:2]),
            "months_collection_instance": int(sample_id[2:4]),
            "mode": int(sample_id[4]),
            "label": int(sample_id[5:8]),
            "sample_num": int(sample_id[8:])
        }

    @staticmethod
    def split_frame(frame, split=(60, 40), shuffle=True):
        """
        split into two dataframes one for training and other for testing
        describes train-test partition of data. (the default is (60, 40))
        the sum of split can be less than 100
        TODO: samples per ref point is kept consistent
        Parameters
        ----------
        frame : Pandas Data frame
        dataframe will be split

        split : tuple    
        """

        # make two empty copies of raw data frame
        # one for test and the other for train
        train = pd.DataFrame(columns=frame.columns)
        test = pd.DataFrame(columns=frame.columns)

        # unique reference points (rps)
        rps = frame[["LABEL"]].drop_duplicates().values.flatten()

        # for each location
        for lbl in rps:
            # get the indices of rows that represent this location
            indices = list(frame.loc[frame["LABEL"] == lbl].index)

            if shuffle:
                np.random.shuffle(indices)

            # get the indices of raw data rows for train and test
            first_split_pos = int(len(indices) * (split[0] / 100))
            second_split_pos = int(len(indices) * (split[1] / 100))

            # use the split positions above to identify splits
            train_indices = indices[0: first_split_pos]
            test_indices = indices[-second_split_pos:]

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
