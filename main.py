from user_config import *
import tensorflow as tf
import keras.backend as K
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import os
from config import mask_config
from model import MaskRCNN

sess = tf.Session()
K.set_session(sess)


def get_config():
    if is_coco:
        import coco
        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

    else:
        config = mask_config(NUMBER_OF_CLASSES)

    return config


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def freeze_model(model, name):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = PATH_TO_SAVE_FROZEN_PB
    tf.train.write_graph(frozen_graph, directory, name , as_text=False)
    print("*"*80)
    print("Finish converting keras model to Frozen PB")
    print('PATH: ', PATH_TO_SAVE_FROZEN_PB)
    print("*" * 80)


def make_serving_ready(model_path, save_serve_path, version_number):
    import tensorflow as tf

    export_dir = os.path.join(save_serve_path, str(version_number))
    graph_pb = model_path

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        input_image = g.get_tensor_by_name("input_image:0")
        input_image_meta = g.get_tensor_by_name("input_image_meta:0")
        input_anchors = g.get_tensor_by_name("input_anchors:0")

        output_detection = g.get_tensor_by_name("mrcnn_detection/Reshape_1:0")
        output_mask = g.get_tensor_by_name("mrcnn_mask/Reshape_1:0")

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input_image": input_image, 'input_image_meta': input_image_meta, 'input_anchors': input_anchors},
                {"mrcnn_detection/Reshape_1": output_detection, 'mrcnn_mask/Reshape_1': output_mask})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save()
    print("*" * 80)
    print("FINISH CONVERTING FROZEN PB TO SERVING READY")
    print("PATH:", PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL)
    print("*" * 80)


# Load Mask RCNN config
# you can also load your own config in here.
# config = your_custom_config_class
config = get_config()


# LOAD MODEL
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(H5_WEIGHT_PATH, by_name=True)

# Converting keras model to PB frozen graph
freeze_model(model.keras_model, FROZEN_NAME)

# Now convert frozen graph to Tensorflow Serving Ready
make_serving_ready(os.path.join(PATH_TO_SAVE_FROZEN_PB, FROZEN_NAME),
                     PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL,
                     VERSION_NUMBER)

print("COMPLETED")