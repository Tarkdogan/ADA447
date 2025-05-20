#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
from fastai.vision.all import *
import pathlib
import os

#load the exported model
learn = load_learner("fruit_classifier.pkl")

#Define prediction model
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return f"Prediction: {pred} (Confidence: {probs[pred_idx]:.4f})"

#Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a fruit (Only orange, banana or apple)"),
    outputs="text",
    title="Ada447-Midterm-Fruit Classifier",
    description="Upload an fruit image. Only banana, apple or orange please")

if __name__ == "__main__":
    interface.launch()


# In[ ]:




