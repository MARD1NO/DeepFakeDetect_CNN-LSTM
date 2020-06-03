# DeepFakeDetect_CNN-LSTM
this is a CNN+LSTM Model to detect deepfake Video
the data is from Kaggle DFDC
97% accuracy in train dataset, 80.7% accuracy in eval dataset

# architecture
- EfficientNet.py I use EfficientNet as backbone
- model_utils.py some operation used in EfficientNet.py
- SaveFrameImage.py use cv2.VideoCapture to capture the frame image into a dir
- SaveFaceImage.py I use paddlehub pretrained face detect model, detect the face in frameimage and crop into a dir
- testScript.py Some faces may not be detected, testScript.py will remove the face imagedir which number less than 10
- FaceDataLoader.py a batched generator to load image data
- convlstm.py use conv to build a lstm-unit
- CNNRNNModel.py build CNN-RNN Model
- face_trainv3.py to train a model

# train
to train a model
```
python face_trainv3.py
```

# eval
remember to choose your eval model name
like
```
python eval.py --weight_file=./DpDcModel_LSTM2_epoch50
```
