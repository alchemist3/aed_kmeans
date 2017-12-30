from model import *

# Iris dataset analysis
print("Iris dataset analysis:")
x, y, meta = load_arff('iris.arff')
# plot_data_set_2d(x, y)
# plot_data_set_3d(x, y)
C, CX = k_means(x, 3)
# plot_clusters_2d(CX, x)
# plot_clusters_3d(CX, x)
error = clustering_error(x, C, CX)
print(f"Clustering error: {round(error,3)}")
CX = find_corresponding_classes(CX, y)
accuracy = kmeans_accuracy(CX, y)
print(f"Accuracy of clustering: {round(accuracy, 4)}\n")

# Seeds dataset analysis:
print("Seeds dataset analysis")
x = [[float(i) for i in line.rstrip('\n').split("\t")[:7]] for line in open("seeds_dataset.txt")]
y = [int(line.rstrip('\n').split("\t")[7]) for line in open("seeds_dataset.txt")]
# plot_data_set_2d(x, y)
# plot_data_set_3d(x, y)
C, CX = k_means(x, 3)
# plot_clusters_2d(CX, x)
# plot_clusters_3d(CX, x)
error = clustering_error(x, C, CX)
print(f"Clustering error: {round(error,3)}")
CX = find_corresponding_classes(CX, y)
accuracy = kmeans_accuracy(CX, y)
print(f"Accuracy of clustering: {round(accuracy, 4)}\n")

# # Banknote_authentication dataset analysis:
print("banknote authentication dataset analysis")
x = [[float(i) for i in line.rstrip('\n').split(",")[:4]] for line in open("banknote_authentication_set.txt")]
y = [int(line.rstrip('\n').split(",")[4]) for line in open("banknote_authentication_set.txt")]
# plot_data_set_2d(x, y)
# plot_data_set_3d(x, y)
C, CX = k_means(x, 2)
# plot_clusters_2d(CX, x)
# plot_clusters_3d(CX, x)
error = clustering_error(x, C, CX)
print(f"Clustering error: {round(error,3)}")
CX = find_corresponding_classes(CX, y)
accuracy = kmeans_accuracy(CX, y)
print(f"Accuracy of clustering: {round(accuracy, 4)}\n")


# # Skin/nonskin dataset analysis:
# print("Skin/nonskin dataset analysis")
# x = [[float(i) for i in line.rstrip('\n').split("\t")[:3]] for line in open("skin_nonskin_dataset.txt")]
# # y = [int(line.rstrip('\n').split("\t")[3]) for line in open("skin_nonskin_dataset.txt")]
# num_of_samples = len(x)
# measured_time = measure_time(x)
# x = np.array([i / 20*num_of_samples for i in range(1, 20)])
# y = measured_time
# model = linear_model.LinearRegression(fit_intercept=True)
# model.fit(x[:, np.newaxis], y)
# xfit = np.linspace(0, num_of_samples, 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y, c='r')
# plt.plot(xfit, yfit)
# plt.title('Zależność czasu wykonania od rozmiaru próbki, model liniowy')
# plt.xlabel('Rozmiar próbki')
# plt.ylabel('Czas wykonania [s]')
# plt.show()
# y_pred = model.predict(x[:, np.newaxis])
# R2 = r2_score(y,y_pred)
# print(f"Coefficient of determination: {round(R2, 4)}\n")
