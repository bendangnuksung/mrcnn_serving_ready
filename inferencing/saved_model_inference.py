import cv2, grpc, sys, os
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import numpy as np
import tensorflow as tf
from inferencing import saved_model_config
from inferencing.saved_model_preprocess import ForwardModel

host = saved_model_config.HOST_NO
port = saved_model_config.PORT_NO

channel = grpc.insecure_channel(str(host) + ':' + str(port), options=[('grpc.max_receive_message_length', saved_model_config.GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


request = predict_pb2.PredictRequest()
request.model_spec.name = saved_model_config.MODEL_NAME
request.model_spec.signature_name = saved_model_config.SIGNATURE_NAME

model_config = saved_model_config.MY_INFERENCE_CONFIG
preprocess_obj = ForwardModel(model_config)


def detect_mask_single_image(image):
    images = np.expand_dims(image, axis=0)
    molded_images, image_metas, windows = preprocess_obj.mold_inputs(images)
    molded_images = molded_images.astype(np.float32)
    image_metas = image_metas.astype(np.float32)
    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = preprocess_obj.get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (images.shape[0],) + anchors.shape)

    request.inputs[saved_model_config.INPUT_IMAGE].CopyFrom(
        tf.contrib.util.make_tensor_proto(molded_images, shape=molded_images.shape))
    request.inputs[saved_model_config.INPUT_IMAGE_META].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_metas, shape=image_metas.shape))
    request.inputs[saved_model_config.INPUT_ANCHORS].CopyFrom(
        tf.contrib.util.make_tensor_proto(anchors, shape=anchors.shape))

    result = stub.Predict(request, 60.)
    result_dict = preprocess_obj.result_to_dict(images, molded_images, windows, result)[0]
    return result_dict


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to Image', required=True)
    args = vars(parser.parse_args())
    image_path = args['path']

    if not os.path.exists(image_path):
        print(f"{image_path}  -- Does not exist")
        exit()

    image = cv2.imread(image_path)
    if image is None:
        print("Image path is not proper")
        exit()

    result = detect_mask_single_image(image)
    print("*" * 60)
    print("RESULTS:")
    print(result)
    print("*" * 60)

