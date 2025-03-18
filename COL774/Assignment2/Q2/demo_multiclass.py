import svm
import helper
import analysis

model = svm.SupportVectorMachine()

path = "../data/Q2/"
array_list_train, array_list_test = helper.read_images(path)

analysis.predict_multiclass(array_list_train, array_list_test)
