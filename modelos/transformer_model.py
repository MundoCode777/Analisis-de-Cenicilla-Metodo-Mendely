# transformer_model.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1,1,1,1], padding="VALID")
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch, -1, patch_dims])

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.position_embedding.input_dim, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerModel:
    def __init__(self):
        self.img_size = (128,128)
        self.patch_size = 16
        self.num_patches = (self.img_size[0]//self.patch_size)**2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_layers = 4
        self.mlp_head_units = [128,64]
        self.is_trained = False
        self.class_names = {
            1:"Clase 1 - Resistente",2:"Clase 2 - Moderadamente tolerante",
            3:"Clase 3 - Ligeramente tolerante",4:"Clase 4 - Susceptible",
            5:"Clase 5 - Altamente susceptible"}
        self.load_or_create_model()

    def load_labels(self):
        p="data/labels.json"
        if not os.path.exists(p): return {}
        with open(p,"r",encoding="utf-8") as f: data=json.load(f)
        return {k:int(v) for k,v in data.items()}

    def create_vit(self):
        inputs=layers.Input(shape=(*self.img_size,3))
        x=layers.Rescaling(1./255)(inputs)
        x=Patches(self.patch_size)(x)
        x=PatchEncoder(self.num_patches,self.projection_dim)(x)
        for _ in range(self.transformer_layers):
            x1=layers.LayerNormalization(epsilon=1e-6)(x)
            att=layers.MultiHeadAttention(num_heads=self.num_heads,key_dim=self.projection_dim,dropout=0.1)(x1,x1)
            x2=layers.Add()([att,x])
            x3=layers.LayerNormalization(epsilon=1e-6)(x2)
            x3=layers.Dense(self.projection_dim*2,activation=tf.nn.gelu)(x3)
            x3=layers.Dropout(0.1)(x3)
            x3=layers.Dense(self.projection_dim,activation=tf.nn.gelu)(x3)
            x3=layers.Add()([x3,x2])
            x=x3
        x=layers.LayerNormalization(epsilon=1e-6)(x)
        x=layers.GlobalAveragePooling1D()(x)
        x=layers.Dropout(0.3)(x)
        for u in self.mlp_head_units:
            x=layers.Dense(u,activation=tf.nn.gelu)(x)
            x=layers.Dropout(0.3)(x)
        outputs=layers.Dense(5,activation="softmax")(x)
        m=keras.Model(inputs,outputs)
        m.compile(optimizer=keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        return m

    def load_images(self,labels):
        X,Y=[],[]
        for n,v in labels.items():
            try:
                img=Image.open(os.path.join("data",n)).convert("RGB").resize(self.img_size)
                X.append(np.array(img,dtype=np.float32))
                Y.append(int(v)-1)
            except: pass
        return np.array(X),np.array(Y)

    def create_and_train(self,labels):
        if len(labels)<20:
            print("❌ Transformer: mínimo 20 imágenes.")
            return False
        X,Y=self.load_images(labels)
        Xtr,Xv,Ytr,Yv=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
        self.model=self.create_vit()
        cb=[callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True,verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=7,verbose=1)]
        print("\n🎓 Entrenando Vision Transformer...")
        self.model.fit(Xtr,Ytr,validation_data=(Xv,Yv),epochs=80,batch_size=16,verbose=1,callbacks=cb)
        val_loss,val_acc=self.model.evaluate(Xv,Yv,verbose=0)
        print(f"✅ Precisión validación: {val_acc:.2%}")
        os.makedirs("modelos",exist_ok=True)
        self.model.save("modelos/transformer_model.h5")
        with open("modelos/transformer_metadata.json","w") as f:
            json.dump({"val_accuracy":float(val_acc),"samples":len(X)},f,indent=2)
        self.is_trained=True
        return True

    def load_or_create_model(self):
        p="modelos/transformer_model.h5"
        labels=self.load_labels()
        if os.path.exists(p):
            try:
                self.model=keras.models.load_model(p,custom_objects={"Patches":Patches,"PatchEncoder":PatchEncoder})
                self.is_trained=True
                print("✅ Transformer cargado correctamente.")
                return
            except Exception as e: print("⚠️ Error cargando modelo:",e)
        if len(labels)>=20: self.create_and_train(labels)
        else: print("❌ Transformer: No hay suficientes imágenes etiquetadas (mínimo 20).")

    def preprocess(self,path):
        img=Image.open(path).convert("RGB").resize(self.img_size)
        arr=np.expand_dims(np.array(img,dtype=np.float32),0)
        return arr

    def predict_image(self,path):
        if not self.is_trained:
            return {"class":-1,"class_name":"Modelo no entrenado","confidence":0.0}
        arr=self.preprocess(path)
        pred=self.model.predict(arr,verbose=0)[0]
        c=int(np.argmax(pred))
        conf=float(pred[c])
        return {
            "class":c+1,
            "class_name":self.class_names[c+1],
            "confidence":conf,
            "probabilities":{self.class_names[i+1]:float(p) for i,p in enumerate(pred)},
            "model":"Transformer"
        }

    def analyze_dataset(self,folder="data"):
        imgs=[f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff"))]
        results=[self.predict_image(os.path.join(folder,f))|{"image_name":f} for f in imgs]
        print(f"✅ Transformer: Analizadas {len(results)} imágenes.")
        return results
