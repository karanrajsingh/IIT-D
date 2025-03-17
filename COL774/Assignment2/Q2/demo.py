import svm

model = svm.SupportVectorMachine()

path = "../data/Q2/"
model.read_images(path)

# model.predict_multiclass()
# model.predict_multiclass_gaussian()
# model.cross_validation()