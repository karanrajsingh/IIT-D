import gda
model = gda.GaussianDiscriminantAnalysis()
X, y = model.read(path_X = './../data/Q4/q4x.dat', path_Y = './../data/Q4/q4y.dat')
same = model.fit(X, y, True)
diff = model.fit(X, y, False)
#print(same, diff)
y_p = model.predict(X[45 : 55], y)