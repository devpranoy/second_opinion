# importing packages
from flask import Flask ,render_template, redirect, url_for, session, request, logging
#import requests
#import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

from werkzeug import secure_filename
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(2048, 2, bias=True)

fc_parameters = model.fc.parameters()

for param in fc_parameters:
    param.requires_grad = True
model.load_state_dict(torch.load('checkpoints.pt',map_location='cpu'))
criterion = nn.CrossEntropyLoss()
def load_input_image(img_path):    
    image = Image.open(img_path)
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image

def predict_malaria(model, class_names, img_path):
    # load the image and return the predicted breed
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]



from glob import glob
from PIL import Image
from termcolor import colored


class_names=['Parasitized','Uninfected']


app = Flask(__name__) #app initialisation

@app.route('/get_label', methods=['GET','POST']) #landing page intent
def label():
    if request.method=='POST':
        title = request.form['title']
        remark = request.form['remark']
        app.config['UPLOAD_FOLDER']="/home/ubuntu/second_opinion/static/uploads/"
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        try:
            f = request.files['file']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
            path = app.config['UPLOAD_FOLDER']+'/'+f.filename
    img_path = np.array(glob(path))
    img = Image.open(img_path)
    if predict_malaria(model, class_names, img_path) == 'Parasitized':
        return "Infected"
    else:
        return "Un-Infected"

@app.route('/', methods=['GET','POST'])
    return render_template("index.html") #display the html template

if __name__=='__main__':
	app.run(debug=True,host="0.0.0.0",port=80) 
    #use threaded=True instead of debug=True for production
    # use port =80 for using the http port



#sample code for form data recieve
# request.form['name']
# Sample Code for JSON send data to api

#url = 'URL_FOR_API'
#data = {'TimeIndex':time1 ,'Name':name,'PhoneNumber':phone}
#headers = {'content-type': 'application/json'}
#r=requests.post(url, data=json.dumps(data), headers=headers)
#data = r.json()
#print(data)


#Sample code for JSON recieve data from API

#url = 'URL_FOR_API'
#headers = {'content-type': 'application/json'}
#r=requests.get(url, headers=headers)
#data = r.json()
#count = data['Count']
