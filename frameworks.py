"""
functions to quickly train and test localization techniques
"""
from numpy.lib.twodim_base import tri
from helpers import get_visible_waps, make_images, split_frame
from seth.Mapping.Floorplan import Floorplan
from seth.Seth import MAC_RE, Devices, fetch_seth
from stone import (
    TripletManager as StoneTripletManager,
    embedding_model as stone_embedding_model,
    complete_model as stone_complete_model,
    pick_negative_1d as stone_pick_negative_1d,
)
from stone.siamese import load_encoder as stone_load_encoder
from paris import (
    TripletManager as ParisTripletManager,
    embedding_model as paris_embedding_model,
    complete_model as paris_complete_model,
    pick_negative_1d as paris_pick_negative_1d,
)
from paris.siamese import load_encoder, save_encoder
from paris.attention import (
    ParisAttention as ParisAttention,
    ParisMultiHeadAttention as ParisMultiHeadAttention,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.callbacks import TQDMProgressBar
from lt import LTKNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from data_helper import get_aps_generic as get_aps
from lt import LTKNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from data_helper import get_aps_generic as get_aps
from tensorflow.keras.models import load_model
import pandas as pd
from typing import List
import tensorflow as tf
from tensorflow_addons.callbacks import TQDMProgressBar



# ######################################
# Stone
# ######################################
def stone_train(train_df,
                val_df=None,
                target=["label"],
                dim_embed=3,
                input_shape=(18, 18, 1),
                learning_rate=1e-4,
                alpha=0.50,
                batch_size=32,
                steps_per_epoch=100,
                p_turn_off=0.80,
                contrast_range=None,
                brightness_delta=None,
                gaussian_noise=0.10,
                model_layers=[50, 50, 100],
                epochs=100,
                callback_loss_patience=20,
                fit_verbose=0,
                nn=1,
                val_bs=100,
                encoder_path=None):

    if encoder_path is not None:
        triplet_encoder, meta = stone_load_encoder(encoder_path)
        train_waps = list(meta["TRAIN_WAPS"])
    else:
        # get train aps
        train_waps = get_aps(train_df.columns)

    if val_df is None:
        train_df, val_df = split_frame(train_df)

    # set up train data
    train_x = (train_df[train_waps].values + 100) / 100
    train_x = make_images(train_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    train_y = train_df[target].values.reshape((-1)).astype(int)

    # setup val data
    # set up train data
    val_x = (val_df[train_waps].values + 100) / 100
    val_x = make_images(val_x.astype(float), force_shape=input_shape[:2])
    # set outputs
    val_y = val_df[target].values.reshape((-1)).astype(int)

    if encoder_path is None:

        triplet_encoder = stone_embedding_model(input_shape,
                                                dim_embed,
                                                gaussian_noise=gaussian_noise,
                                                model_layers=model_layers)

        # put the encoder into the stone system
        siamese = stone_complete_model(triplet_encoder, input_shape, learning_rate, alpha)

        # setup data generators and feed the monster!
        train_gen = StoneTripletManager(train_x,
                                        train_y,
                                        n_sampler=stone_pick_negative_1d,
                                        steps_per_epoch=steps_per_epoch,
                                        p_turn_off=p_turn_off,
                                        contrast_range=contrast_range,
                                        brightness_delta=brightness_delta,
                                        bs=batch_size)

        # validation data generator
        val_gen = StoneTripletManager(val_x,
                                      val_y,
                                      n_sampler=stone_pick_negative_1d,
                                      steps_per_epoch=steps_per_epoch,
                                      p_turn_off=p_turn_off,
                                      contrast_range=contrast_range,
                                      brightness_delta=brightness_delta,
                                      bs=val_bs)

        # fit the model
        history = siamese.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              verbose=fit_verbose,
                              callbacks=[
                                  EarlyStopping(monitor="val_loss",
                                                patience=callback_loss_patience,
                                                restore_best_weights=True),
                                  TQDMProgressBar(show_epoch_progress=False,
                                                  leave_overall_progress=False)
                              ])

        print("Loss:", history.history["loss"][-1])

    predictor = KNN(nn)
    # NOTE: Is it better to train using Augmented Data?
    train_encodings = triplet_encoder.predict(train_x)
    predictor.fit(train_encodings, train_y.flatten())

    return triplet_encoder, predictor, train_waps


def stone_test(test_df, encoder, predictor, train_waps, input_shape=(18, 18, 1)):

    # any waps mising can be fixed
    test_waps = get_aps(list(test_df.columns))
    missing_waps = set(train_waps) - set(test_waps)
    test_df[list(missing_waps)] = -100

    # tx
    tx = np.array(test_df[train_waps].values, dtype=np.float)
    tx = (tx + 100) / 100
    tx = make_images(tx, force_shape=input_shape[:2])

    # test_y = np.array(test_df["label"].values, dtype=np.int).flatten()
    # test_y = test_y.reshape((-1, 1))

    # predict test using "train_waps"
    encoded = encoder.predict(tx)
    return predictor.predict(encoded).flatten()


# #####################################


