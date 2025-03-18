import svm
import helper
import analysis
import numpy as np

model = svm.SupportVectorMachine()

path = "../data/Q2/"
array_list_train, array_list_test = helper.read_images(path)

X_train, y_train, X_test, y_test = helper.build_input(array_list_train, array_list_test, class_index_0 = 0, class_index_1= 1)
y_train_adj = (y_train + 1)/2
y_test_adj =  (y_test + 1)/2

# Part 4
print("LIBLINEAR")
analysis.fit_liblinear(X_train, y_train, X_test, y_test, C = 1.0)
print("LIBLINEAR ends...")
print("SGD")
analysis.fit_sgd(X_train, y_train, X_test, y_test)
print("SGD ends...")

# Part 3 linear case, Part 1 (uncomment sort_and_plot in fit)
print("CVXOPT LINEAR")
model.fit(X_train, y_train_adj, kernel='linear')
y_predicted = model.predict(X_test)
accuracy = model.accuracy(y_test_adj, y_predicted)
w, b, support_vectors_cvx_lin, X_supp_cvx_lin, support_indices_cvx_lin = model.get_params()
analysis.sort_and_plot(X_supp_cvx_lin, support_vectors_cvx_lin, w)
print("Accuracy: ", accuracy)
print("CVXOPT LINEAR ends...")

# Part 3 linear case (uncomment weights and intercept inside fit_svc)
print("SVC LINEAR")
support_indices_svc_lin = analysis.fit_svc(X_train, y_train, X_test, y_test, kernel='linear')
print("SVC LINEAR ends...")

# Part 3 gaussian case, Part 2 (uncomment sort_and_plot in fit)
print("CXVOPT GAUSSIAN")

support_indices_cvx_gau = model.fit(X_train, y_train_adj, kernel='gaussian')
y_predicted = model.predict(X_test)
accuracy = model.accuracy(y_test_adj, y_predicted)
support_vectors_cvx_gau, X_supp_cvx_gau, support_indices_cvx_gau = model.get_params()
analysis.sort_and_plot_gaussian(X_supp_cvx_gau, support_vectors_cvx_gau)
print("Accuracy: ", accuracy)

# Part 3 gaussian case
print("SVC GAUSSIAN")
support_indices_svc_gau = analysis.fit_svc_gaussian(X_train, y_train, X_test, y_test, kernel='rbf')

print("Common cvx: ", len(np.intersect1d(np.array(support_indices_cvx_gau), np.array(support_indices_cvx_lin))))
print("Common svc: ", len(np.intersect1d(np.array(support_indices_svc_gau), np.array(support_indices_svc_lin))))
print("Common lin: ", len(np.intersect1d(np.array(support_indices_svc_lin), np.array(support_indices_cvx_lin))))
print("Common gau: ", len(np.intersect1d(np.array(support_indices_svc_gau), np.array(support_indices_cvx_gau))))
