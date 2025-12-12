from ultralytics import YOLO
import os

data_yaml = 'data.yaml'  

model_name = 'yolov12x.pt' 

model = YOLO(model_name)

# Entrenar el modelo
results = model.train(
    data=data_yaml,          
    epochs=100,              
    imgsz=640,                
    batch=16,                 
    name='yolov12x_custom',    
    device=0,                 
    workers=4,                
    patience=30,              
    save_period=10,           
    augment=True,            
    close_mosaic=10,         
    project='runs',           
    exist_ok=True             
)

print("Entrenamiento completado.")
print("Resultados guardados en: runs/train")
