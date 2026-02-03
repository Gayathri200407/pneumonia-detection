from fastai.vision.all import *

path = Path('data/chest_xray')

data = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='val',
    item_tfms=Resize(224)
)

learn = cnn_learner(data, resnet18, metrics=accuracy)
learn.fine_tune(4)

learn.export('pneumonia_model.pkl')
