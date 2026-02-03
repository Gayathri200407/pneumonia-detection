from fastai.vision.all import *

learn = load_learner('pneumonia_model.pkl')
img = PILImage.create('test_image.jpg')
pred, _, probs = learn.predict(img)

print(f"Prediction: {pred}")
print(f"Probability: {max(probs):.4f}")
