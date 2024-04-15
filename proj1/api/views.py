from flask import Blueprint , jsonify ,request
# from . import db
# from .model import pred_data
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
import keras.backend as K

main = Blueprint('main', __name__)
# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)   
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


custom_objects = {'attention': attention}
# model = pickle.load(open(r'C:/Users/HP/Documents/BE-Model/model.h5','rb'))
model1 = load_model(r'E:\BE project\proj1\api\model.h5', custom_objects=custom_objects)
# scl = pickle.load(open(r'C:\Users\HP\Desktop\Hackathon challenge\weather\slv1.sav','rb'))
scl = pickle.load(open(r'E:\BE project\proj1\api\tk.pkl','rb'))
# wght = open(r"C:/Users/HP/Documents/BE-Model/model_weights.h5", "r")


@main.route('/predict1',methods=['POST','GET'] )
def predict_disease():

    data = request.get_json(force=True)
   # console.log(data)
    data = data.values()
    sd = scl.texts_to_sequences(data)
    data = pad_sequences(sd, padding='post', maxlen=234)
    pd = model1.predict([data])
    #new_value = pred_data(name='tanush',value=pd)
    # db.session.add(new_value)
    # db.session.commit()
    result = "safe"
    if pd[0][0] >= 0.4 :
        result = "unsafe"
    return jsonify({"values": result}),200

@main.route('/retrieve',methods=['POST','GET'] )
def predict_disease1():
    data1 = request.get_json(force=True)
    x = data1['name']
   # dis_list = pred_data.query.filter_by(name=x).all()
    ret_list = []
   # for x in dis_list:
   #    ret_list.append({"value":x.value})
    return jsonify({"values":ret_list})


    
  #  pd = Pred_data(city=data['city'],Temp=data['Temp'],Hum=data['Hum'],Prp=data['Prp'],Ws=data['Ws']) 
  #  jsonify({'value' : int(pd)})
    
    
      
