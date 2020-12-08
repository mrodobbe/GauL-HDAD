from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties

# TODO: Connect with other files

try:
    file_location = sys.argv[1]
    # validation_file = sys.argv[2]
except:
    print("Not enough input files given.")
    raise

with open(str(file_location + "/training_intermediate.pickle"), "rb") as f:
    training_representations = pickle.load(f)
training_representations = np.asarray(training_representations)
print("Loaded the training representations!")

with open(str(file_location + "/validation_intermediate.pickle"), "rb") as f:
    validation_representations = pickle.load(f)
validation_representations = np.asarray(validation_representations)
print("Loaded the validation representations!")
with open(str(file_location + "/test_intermediate.pickle"), "rb") as f:
    test_representations = pickle.load(f)
test_representations = np.asarray(test_representations)
all_representations = np.concatenate((training_representations, validation_representations, test_representations))
# training_features = np.zeros(len(training_representations)) - 4
training_features = np.loadtxt(str(file_location + "/training_enthalpies.txt")).astype(np.float)
# validation_features = np.loadtxt(str(file_location + "/validation_features.txt")).astype(np.float)
validation_features = np.loadtxt(str(file_location + "/validation_enthalpies.txt")).astype(np.float)

test_features = np.loadtxt(str(file_location + "/test_features.txt")).astype(np.float)
features = np.append(training_features, validation_features)
n = len(features)

X_centered = all_representations - all_representations.mean(axis=0)
X_normalized = X_centered / sum(all_representations.mean(axis=0))
print(all_representations.mean(axis=0))

# pca = PCA(n_components=3)
# X_embedded = pca.fit_transform(all_representations)
# print(pca.explained_variance_ratio_)

svd = TruncatedSVD(n_components=9)
X = svd.fit_transform(X_normalized)
# X = svd.fit_transform(all_representations)
X_embedded = TSNE(n_components=2, perplexity=35, n_iter=10000).fit_transform(X)
# X_embedded = TSNE(perplexity=20, n_iter=15000).fit_transform(all_representations)

# train_embedded = np.loadtxt(str(file_location + "/training_coordinates.txt"))
# test_embedded = np.loadtxt(str(file_location + "/test_coordinates.txt"))
# X_embedded = np.concatenate((train_embedded, test_embedded))
ln_largest_error = max(test_features)
largest_error = int(np.floor(np.exp(ln_largest_error)))
print(largest_error)

fig = plt.figure()
ax1 = fig.add_subplot(111)
hfont = {'fontname': 'UGent Panno Text'}
font = FontProperties(family='UGent Panno Text',
                      weight='normal',
                      style='normal', size=15)
plt.rc('font', size=16)
h = ax1.scatter(X_embedded[:n, 0], X_embedded[:n, 1], c=features, cmap='RdBu_r', edgecolors="black",
                label="Training Molecules")
c_bar = plt.colorbar(h, pad=0.005)
c_bar.ax.set_ylabel('Enthalpy of Formation [kJ mol$^{-1}$]', rotation=270, size=18, labelpad=15, **hfont)
t = ax1.scatter(X_embedded[n:, 0], X_embedded[n:, 1], c=test_features, cmap='Greens',
                edgecolors="#0d983a", marker="^", s=80, label="Test Molecules")
err_ax = inset_axes(ax1, width="35%", height="3%", loc=3, bbox_to_anchor=(0, 0.03, 1, 1),
                    bbox_transform=ax1.transAxes)
err = plt.colorbar(t, cax=err_ax, ticks=[0., np.log(10), np.log(50), np.log(100),
                                         ln_largest_error], orientation='horizontal')
err.ax.set_xticklabels([0, 10, 50, 100, largest_error])
err.ax.set_xlabel("Absolute Deviation Test Point [kJ mol$^{-1}$]", size=14, **hfont)
err.ax.xaxis.set_ticks_position("top")
h.axes.get_xaxis().set_visible(False)
h.axes.get_yaxis().set_visible(False)
for tick in err.ax.get_xticklabels():
    tick.set_fontname("UGent Panno Text")
for tick in c_bar.ax.get_yticklabels():
    tick.set_fontname("UGent Panno Text")
plt.legend((h, t), ('Training molecules', 'Test molecules'), loc='lower left', bbox_to_anchor=(0, 0.15),
           bbox_transform=ax1.transAxes, prop=font)
plt.show()

# np.savetxt(str(file_location + "/training_coordinates_full.txt"), X_embedded[:n, :], fmt='%.4f')
# np.savetxt(str(file_location + "/test_coordinates_full.txt"), X_embedded[n:, :], fmt='%.4f')
