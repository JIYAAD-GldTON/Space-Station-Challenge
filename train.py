from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="yolo_parameters.yaml",  # Path to dataset configuration file
    epochs=10,  # Number of training epochs
    imgsz=320,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

metrics = model.val()
path = model.export(format="onnx")