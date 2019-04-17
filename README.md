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
