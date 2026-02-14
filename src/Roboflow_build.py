from roboflow import Roboflow


print("Connecting to workspace...")
project = rf.workspace("object-detection-using-roboflow-and-active-learning").project("dice-03h7s")
print("Project loaded, downloading dataset...")
version = project.version(2)
dataset = version.download("yolov8")
print(f"Dataset saved to: {dataset.location}")