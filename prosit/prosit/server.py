import os
import tempfile
import warnings
import flask
from flask import after_this_request
import pandas as pd
import tensorflow as tf

from . import model
from . import io_local
from . import constants
from . import tensorize
from . import prediction
from . import alignment
from . import converters

## for training
from . import losses
from . import model as model_lib
from . import constants


app = flask.Flask(__name__)


@app.route("/")
def hello():
    return "prosit!\n"


def predict(csv):
    df = pd.read_csv(csv)
    data = tensorize.csv(df)
    data = prediction.predict(data, d_spectra)
    data = prediction.predict(data, d_irt)
    return data


def get_callbacks(model_dir_path):
    import keras

    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:02d}"
    weights_file = "{}/weight_{}_{}.hdf5".format(
        model_dir_path, epoch_format, loss_format
    )
    save = keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True)
    stop = keras.callbacks.EarlyStopping(patience=10)
    decay = keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2)
    return [save, stop, decay]


def train(tensor, model, model_config, callbacks):
    import keras

    if isinstance(model_config["loss"], list):
        loss = [losses.get(l) for l in model_config["loss"]]
    else:
        loss = losses.get(model_config["loss"])
    optimizer = model_config["optimizer"]
    x = io_local.get_array(tensor, model_config["x"])
    y = io_local.get_array(tensor, model_config["y"])
    
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(
        x=x,
        y=y,
        epochs=constants.TRAIN_EPOCHS,
        batch_size=constants.TRAIN_BATCH_SIZE,
        validation_split=1 - constants.VAL_SPLIT,
        callbacks=callbacks,
    )
    keras.backend.get_session().close()

@app.route("/train/toppeaks", methods=["GET"])
def return_train():
    print(" TRAIN TOP-PEAKS-3 ")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # turn off tf logging
    data_path = constants.DATA_PATH
    model_dir = constants.MODEL_DIR

    model, model_config = model_lib.load(model_dir, trained=False)
    tensor = io_local.from_hdf5(data_path)
    callbacks = get_callbacks(model_dir)
    train(tensor, model, model_config, callbacks)
    model_lib.save(model, model_config, constants.OUT_DIR)

@app.route("/predict/generic", methods=["POST"])
def return_generic():
    result = predict(flask.request.files["peptides"])
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    c = converters.generic.Converter(result, tmp_f.name)
    c.convert()

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)


@app.route("/predict/msp", methods=["POST"])
def return_msp():
    result = predict(flask.request.files["peptides"])
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    c = converters.msp.Converter(result, tmp_f.name)
    c.convert()

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)


@app.route("/predict/msms", methods=["POST"])
def return_msms():
    result = predict(flask.request.files["peptides"])
    df_pred = converters.maxquant.convert_prediction(result)
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    converters.maxquant.write(df_pred, tmp_f.name)

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    global d_spectra
    global d_irt
    d_spectra = {}
    d_irt = {}

    d_spectra["graph"] = tf.Graph()
    with d_spectra["graph"].as_default():
        d_spectra["session"] = tf.Session()
        with d_spectra["session"].as_default():
            d_spectra["model"], d_spectra["config"] = model.load(
                constants.MODEL_SPECTRA,
                trained=True
            )
            print("MODEL SEPCTRA: "+constants.MODEL_SPECTRA)
            d_spectra["model"].compile(optimizer="adam", loss="mse")
    d_irt["graph"] = tf.Graph()
    with d_irt["graph"].as_default():
        d_irt["session"] = tf.Session()
        with d_irt["session"].as_default():
            d_irt["model"], d_irt["config"] = model.load(constants.MODEL_IRT,
                    trained=True)
            d_irt["model"].compile(optimizer="adam", loss="mse")
    app.run(host="0.0.0.0")
