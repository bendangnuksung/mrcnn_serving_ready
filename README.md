### MRCNN Model conversion
Script to convert [MatterPort Mask_RCNN](https://github.com/matterport/Mask_RCNN) Keras model to Tensorflow Frozen Graph and Tensorflow Serving Model.


### How to Run
1. Modify the path variables in 'user_config.py'
2. Run main.py
    ```bash
    python3 main.py
    ```
    
#### For Custom Config class
If you have a different config class you can replace the existing config in 'main.py'
```python
# main.py
# Current config load
config = get_config()

# replace it with your config class
config = your_custom_config_class

```

### Inferencing
Follow once you finish converting it to a `saved_model` using the above code

#### Tensorflow Model Server with GRPC

1. First run your `saved_model.pb` in Tensorflow Model Server, using:
    ```bash
    tensorflow_model_server --port=8500 --model_name=mask --model_base_path=/path/to/saved_model/
    ```
2. Modify the variables and add your Config Class if needed in `inferencing/saved_model_config.py`. No need to change if the saved_model is the default COCO model.
3. Then run the `inferencing/saved_model_inference.py` with the image path:
    ```bash
    # Set Python Path
    export PYTHONPATH=$PYTHONPATH:$pwd
    # Run Inference
    python3 inferencing/saved_model_inference.py -p test_image/monalisa.jpg
    ```
    
 #### Please do send a PR if you know to inference using TF model server RESTAPI.
    
