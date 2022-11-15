"""
Build the stone models for datasets and then save them within the stone module
"""
#
from tensorflow.keras import layers
from tqdm import tqdm
from uji import UJI
import numpy as np
import logging as lg
from helpers import compute_distances, make_images
from sklearn.neighbors import KNeighborsClassifier
# TODO: Merge the three components together
from stone.siamese import TripletManager, embedding_model, complete_model
from stone.resnet_blocks import ResnetConvolutionBlock, ResnetIdentityBlock
from tensorflow import keras
from scipy.stats import multivariate_normal
import tensorflow_addons as tfa
import pickle
lg.basicConfig(format="", level=lg.ERROR)


def pick_negative_uji(
    anchor: int, labels: list,
    sigma=30
):

    # build cov mat
    sigma = np.array([
        [sigma, -0, ],
        [-0,  sigma],
    ])

    # floor mat and label2ij
    floor_mat, label2ij = UJI.uji_floor_mat_and_dict()

    # find the i and j coordinate of label in floor_mat
    i, j = label2ij[anchor]

    # mu should be at anchor
    mu = np.array([i, j])

    # create a multivariate normal
    bi_normal = multivariate_normal(mu, sigma)

    # Note: this might be inverted
    X = np.arange(floor_mat.shape[0])
    Y = np.arange(floor_mat.shape[1])
    X, Y = np.meshgrid(X, Y)
    # pdf calls require this mat
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = bi_normal.pdf(pos)
    # set probability at label to zero
    Z[j, i] = 0.0

    # get probabilities for each label
    label_probs = np.zeros((106,))
    for label, (i, j) in label2ij.items():
        label_probs[label] = Z[j, i]

    label_probs = label_probs/np.sum(label_probs)
    return np.random.choice(np.arange(106), p=label_probs)


def build_uji_dataset(ci=4, floor=3, keep=[],
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


def build_uji_model(trains, tests, ujis,
                    tqdm_disable=False,
                    custom_layers=None):
    keras.backend.clear_session()

    # first month
    months = tuple(trains.keys())
    first_month = months[0]

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
    lr = 1e-4
    epochs = 100
    alpha = 0.2
    nn = 1
    input_shape = train_x.shape[1:]

    # reshape train_x
    accuracies = {}
    pbar = tqdm(months, disable=tqdm_disable)

    # #################
    # setup models
    # #################
    triplet_encoder = embedding_model(
        input_shape,
        embedding_dim,
        custom_layers=custom_layers,
        gaussian_noise=0.10,
        model_layers=[64, 128, 200]
    )

    print(triplet_encoder.summary())

    # put the encoder into the stone system
    siamese = complete_model(triplet_encoder, input_shape, lr, alpha)

    # setup data generators and feed the monster!
    train_gen = TripletManager(train_x, train_y,
                               n_sampler=pick_negative_uji,
                               steps_per_epoch=100,
                               p_turn_off=0.90,
                               bs=batch_size)

    # use month 1 tests as validation data
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
                            bs=1000)


    # set up callbacks
    callbacks = [
                 keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                                 restore_best_weights=True),
                 tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    ]


    # print some details about input output shapes
    [anchor, positive, negative], labels = train_gen.__getitem__(0)

    print(anchor.shape, positive.shape, negative.shape, labels.shape)
    print(anchor.dtype, positive.dtype, negative.dtype, labels.dtype)

    # train the siamese model
    history = siamese.fit(
        train_gen,
        validation_data=val_gen.__getitem__(0),
        epochs=epochs,
        callbacks=callbacks
    )

    # encode everything
    train_encodings = triplet_encoder.predict(train_x)

    # we should really just save this
    siamese_model = KNeighborsClassifier(nn)
    siamese_model.fit(train_encodings, train_y.flatten())

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
        pred_y = siamese_model.predict(encoded)
        pred_y_coord = uji.labels_to_coords(pred_y)

        # compute the distances
        dists = compute_distances(pred_y_coord,
                                  test_y_coord).flatten()

        # store the distances as a 2D array
        accuracies[month] = dists

        pbar.desc = f"{month}: {np.mean(dists):.2f}"

        lg.debug(f"{month} : {np.mean(dists)}")

    return accuracies


def plot_errors_monthly(accuracies, show=False, save=True):

    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 4))
    for tech in accuracies.keys():
        aes = []
        for month in accuracies[tech].keys():

            ae = np.mean(accuracies[tech][month])
            aes.append(ae)

        plt.plot(aes, label=tech, linewidth=3)
        plt.title(f"Temporal Degradation of Localization Performance", size=18)

    plt.xlabel("Month", size=18)
    plt.ylabel("Accuracy (meters)", size=18)

    # get the list of months
    months = list(accuracies[list(accuracies.keys())[0]].keys())
    plt.xticks(ticks=list(range(len(months))),
               labels=months, size=14)

    plt.yticks(size=14)
    plt.legend(fontsize=14, loc="upper left",
               bbox_to_anchor=(1, 0, 1, 1.03))
    plt.tight_layout()

    if save:
        plt.savefig("stone.png")

    if show:
        plt.show()


def main():

    keras.backend.clear_session()

    accuracies = {} 

    trains, tests, ujis = build_uji_dataset(month_range=[1, 2])

    accuracies["stone"] = build_uji_model(trains, tests, ujis,
                                          custom_layers=None)

    plot_errors_monthly(accuracies)

    # print(accuracies["stone"])

if __name__ == "__main__":
    main()
