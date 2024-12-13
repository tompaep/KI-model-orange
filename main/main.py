from ultralytics import YOLO

# Initialisiere ein Modell ohne vortrainierte Gewichte
model = YOLO("../yolo11n.pt")  # Diese Datei definiert die Architektur ohne vortrainierte Gewichte

# Trainiere das Modell mit zufälligen Gewichten
results = model.train(data="config.yaml", epochs=50)
