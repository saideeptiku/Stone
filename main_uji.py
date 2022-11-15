"""
Main file for this project
Each function tests a technique
"""
from tqdm import tqdm
from uji import UJI
import numpy as np
import logging as lg
from helpers import compute_distances, make_images
import tensorflow as tf
from tensorflow import keras
from stone.siamese import (TripletManager, embedding_model,
                          complete_model, pick_negative_uji)
from sklearn.neighbors import KNeighborsClassifier
import json
import pickle
lg.basicConfig(format="", level=lg.INFO)


def stone_test(trains, tests, ujis, tqdm_disable=False, saved_encoder=None, save=True):
    """
    build the siamese model if not available
    """
    tf.keras.backend.clear_session()

    # first month
    months = tuple(trains.keys())
    first_month = 1

    # build the model
    train_mf = trains[first_month]  # get train data for month 1
    train_waps = ujis[first_month].get_visible_waps(records=train_mf)
    (train_x, train_y) = (train_mf[train_waps].values,
                          train_mf[["LABEL"]].values.reshape((-1,)).astype(np.int))
    train_x = make_images(train_x).astype(np.float)
    train_x = (train_x + 100)/100

    # configs for model
    embedding_dim = 36
    batch_size = 500
    lr = 1e-3
    epochs = 20
    alpha = 0.20
    input_shape = train_x.shape[1:]

    if save:

        if saved_encoder is None:
            saved_encoder = "stone/saved_encoders/DEFAULT_MODEL"

        # #################
        # setup models
        # #################
        triplet_encoder = embedding_model(
            input_shape,
            embedding_dim,
            gaussian_noise=0.10,
            model_layers=[64, 128, 200]
        )
        # put the encoder into the stone system
        siamese = complete_model(triplet_encoder, input_shape, lr, alpha)

        # setup data generators and feed the monster!
        train_gen = TripletManager(train_x, train_y,
                                   n_sampler=pick_negative_uji,
                                   steps_per_epoch=100,
                                   p_turn_off=0.99,
                                   bs=batch_size)

        val_m = tests[first_month]
        uji = ujis[first_month]

        # create array of true labels
        vy = val_m[["LABEL"]].values.flatten()

        vx = make_images(val_m[train_waps].values)
        vx = np.array(vx).astype(np.float)
        vx = (vx + 100)/100

        val_gen = TripletManager(vx, vy,
                                 n_sampler=pick_negative_uji,
                                 steps_per_epoch=100,
                                 p_turn_off=0.90,
                                 bs=300)

        # train the siamese model
        history = siamese.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=20,
                                                     restore_best_weights=True)]
        )

        # save the model info
        # save the model
        if save:
            triplet_encoder.save(saved_encoder)
            json.dump({"TRAIN_WAPS": list(train_waps)}, open(
                f"{saved_encoder}/model_info.json", "w"))
    else:
        train_waps = json.load(
            open(f"{saved_encoder}/model_info.json", "r"))["TRAIN_WAPS"]
        triplet_encoder = tf.keras.models.load_model(saved_encoder)

    # encode everything
    train_encodings = triplet_encoder.predict(train_x)

    knn_model = KNeighborsClassifier(1)
    knn_model.fit(train_encodings, train_y.flatten())

    # reshape train_x
    accuracies = {}
    pbar = tqdm(months, disable=tqdm_disable)

    # iterate over each test month
    # go over months beyond the first provided month
    for month in pbar:

        test_cm = tests[month]
        uji = ujis[month]

        # create array of true labels
        test_y_lbls = test_cm[["LABEL"]].values.flatten()
        test_y_coord = uji.labels_to_coords(test_y_lbls)

        tx = make_images(test_cm[train_waps].values)
        tx = np.array(tx).astype(np.float)
        tx = (tx + 100)/100

        # predict test using "train_waps"
        encoded = triplet_encoder.predict(tx)
        pred_y = knn_model.predict(encoded)
        pred_y_coord = uji.labels_to_coords(pred_y)

        # compute the distances
        dists = compute_distances(pred_y_coord,
                                  test_y_coord).flatten()

        # store the distances as a 2D array
        accuracies[month] = dists

        pbar.desc = f"{month}: {np.mean(dists):.2f}"

        lg.debug(f"{month} : {np.mean(dists)}")

    return accuracies


def build_datasets(ci=4, floor=3, keep=[],
                   split=(80, 20), month_range=None):
    trains = {}
    tests = {}
    # stores the UJI objects for each month
    ujis = {}

    lg.info("Building Datasets")
    if month_range is None:
        month_range = list(range(1, 16))

    for i in tqdm(month_range):

        # build the dataset
        uji = UJI.from_cache("test", i, cache_dir="uji/db_cache")

        df = uji.filter_record(
            FLOOR=floor,
        )

        if len(keep) > 0:
            df = df[df["LABEL"].isin(keep)]

        # train and test data split for this month
        train_mi, test_mi = UJI.split_frame(df, split=split)

        # store in dicts
        trains[i] = train_mi
        tests[i] = test_mi
        ujis[i] = uji

    return trains, tests, ujis


def plot_errors_monthly(accuracies, show=True, save=True, fileref="motivation", legend=None):

    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 4))

    if legend is not None:
        keys = legend
    else:
        keys = accuracies.keys()

    for tech in keys:
        aes = []
        for month in accuracies[tech].keys():

            ae = np.mean(accuracies[tech][month])
            aes.append(ae)

        plt.plot(aes, label=tech, linewidth=3)
        # plt.title(f"Temporal Degradation of Localization Performance", size=18)

    plt.xlabel("Month", size=18)
    plt.ylabel("Mean Accuracy (m)", size=18)

    # get the list of months
    months = list(accuracies[list(accuracies.keys())[0]].keys())
    plt.xticks(ticks=list(range(len(months))),
               labels=months, size=14)

    plt.yticks(size=14)
    plt.legend(fontsize=14, loc="upper center",
               ncol=5,
               fancybox=False, shadow=False,
               bbox_to_anchor=(0.5, 1.2)
               )
    plt.tight_layout()

    if save:
        plt.savefig(f"plots/{fileref}.png", dpi=600)

    if show:
        plt.show()


def main():

    trains, tests, ujis = build_datasets(keep=list(range(34)))

    accuracies = {}


    lg.info("Testing STONE")
    accuracies["STONE"] = stone_test(trains, tests, ujis,
                                     saved_encoder="stone/saved_encoders/UJI_M1",
                                     save=False,
                                     tqdm_disable=False)
    lg.debug(accuracies["STONE"])

    plot_errors_monthly(accuracies,
                        fileref="uji_lim34",
                        legend=["STONE"])


if __name__ == "__main__":
    main()
