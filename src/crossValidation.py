import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import Model
import pickle
from src.makeModel import model_builder
from src.makeMolecule import denormalize
from sklearn.svm import SVR
from src.plots import performance_plot
from joblib import wrap_non_picklable_objects


@wrap_non_picklable_objects
def run_cv(all_molecules, all_heavy, x, y, loop, i, save_folder, target, train_val_split=0.9):
    train = loop[0]
    test = loop[1]
    all_molecules = np.asarray(all_molecules)
    training_molecules = all_molecules[train]
    test_molecules = all_molecules[test]

    heavy_train = all_heavy[train]
    heavy_test = all_heavy[test]

    x_train_all = x[train]
    y_train_all = y[train]
    x_test = x[test]
    y_test = y[test]

    n = len(y_train_all)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_val = round(train_val_split * n)
    val_index = indices[n_val:]
    train_index = indices[:n_val]

    x_train = x_train_all[train_index, :]
    y_train = y_train_all[train_index]
    x_val = x_train_all[val_index, :]
    y_val = y_train_all[val_index]
    heavy_val = heavy_train[val_index]

    if len(y.shape) == 2:
        output_layer_size = y.shape[1]
    else:
        output_layer_size = 1

    model = model_builder(x_train_all, output_layer_size)
    model.summary()

    print('{} training molecules, {} validation molecules'.format(len(x_train),
                                                                  len(x_val)))

    es = EarlyStopping(patience=200, restore_best_weights=True, min_delta=0.01, mode='min')

    class LossHistory(Callback):
        def on_epoch_end(self, batch, logs={}):
            print('{:.3f}\t\t-\t{:.3f}'.format(logs.get('val_mean_absolute_error'), np.sqrt(logs.get('val_loss'))))

    lh = LossHistory()

    print('Validation MAE\t-\tValidation MSE')
    history = model.fit(x_train, y_train, epochs=2000,
                        validation_data=(x_val, y_val),
                        batch_size=8, callbacks=[es, lh], verbose=0)
    model.save(str(save_folder + "/Fold {}".format(i)))
    if target != "cp":
        validation_predictions = model.predict(x_val).reshape(-1)
    else:
        validation_predictions = np.asarray(model.predict(x_val)).astype(np.float)

    if target != "h":
        validation_predictions = denormalize(validation_predictions, heavy_val, target, coefficient=1.5)
        y_val = denormalize(y_val, heavy_val, target, coefficient=1.5)

    validation_error = np.abs(validation_predictions - y_val)
    validation_mean_absolute_error = np.average(validation_error)
    validation_root_mean_squared_error = np.sqrt(np.average(validation_error ** 2))

    print('Validation performance statistics:')
    print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(validation_mean_absolute_error))
    print('Root mean squared error:\t{:.2f} kJ/mol'.format(validation_root_mean_squared_error))

    if target != "cp":
        test_predictions = model.predict(x_test).reshape(-1)
    else:
        test_predictions = np.asarray(model.predict(x_test)).astype(np.float)

    if target != "h":
        test_predictions = denormalize(test_predictions, heavy_test, target, coefficient=1.5)
        y_test = denormalize(y_test, heavy_test, target, coefficient=1.5)

    test_error = np.abs(test_predictions - y_test)
    test_mean_absolute_error = np.average(test_error)
    test_root_mean_squared_error = np.sqrt(np.average(test_error ** 2))

    print('Test performance statistics for ANN:')
    print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(test_mean_absolute_error))
    print('Root mean squared error:\t{:.2f} kJ/mol'.format(test_root_mean_squared_error))

    if target != "cp":
        intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('layer_3').output)
        training_intermediates = np.asarray(intermediate_layer(x_train_all)).astype(np.float)
        test_intermediates = np.asarray(intermediate_layer(x_test)).astype(np.float)

        krr = SVR(kernel="rbf", gamma='scale', C=2.5e3)  # This is the support vector machine.
        # Try to find an algorithm that optimizes gamma and C. You can also add an epsilon factor
        krr.fit(training_intermediates, y_train_all)  # Execute regression

        y_svr = krr.predict(test_intermediates)  # Prediction for the test set (unseen data)
        svr_error = np.abs(y_svr - y_test)
        svr_mean_absolute_error = np.average(svr_error)
        svr_root_mean_squared_error = np.sqrt(np.average(svr_error ** 2))

        print('Test performance statistics for SVR:')
        print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(svr_mean_absolute_error))
        print('Root mean squared error:\t{:.2f} kJ/mol'.format(svr_root_mean_squared_error))

    with open(str(save_folder + "/Fold {}/test_results_fold_{}.txt".format(i, i)), "w") as f:
        f.write('ANN Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(validation_mean_absolute_error))
        f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(validation_root_mean_squared_error))
        if target != "cp":
            f.write('SVR Test performance statistics:\n')
            f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(svr_mean_absolute_error))
            f.write('Root mean squared error:\t{:.2f} kJ/mol\n'.format(svr_root_mean_squared_error))
        f.close()

    with open(str(save_folder + "/Fold {}/test_predictions_{}.txt".format(i, i)), "w") as f:
        f.write(str("Molecule \t Real Value \t Prediction \t Absolute Error \n"))
        for m, v, p, e in zip(test_molecules, y_test, test_predictions, test_error):
            f.write(str(m) + '\t' + str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(e, 4)) + '\n')
        f.close()

    with open(str(save_folder + "/Fold {}/test_representations_fold_{}.pickle".format(i, i)), "wb") as f:
        pickle.dump(x_test, f)
    print("Dumped the test molecules!")

    with open(str(save_folder + "/Fold {}/test_outputs_fold_{}.pickle".format(i, i)), "wb") as f:
        pickle.dump(y_test, f)
    print("Dumped the test outputs!")

    performance_plot(y_test, test_predictions, "test", prop=target, folder=str(save_folder + "/Fold {}".format(i)),
                     model="ANN", fold=i)
    if target != "cp":
        performance_plot(y_test, y_svr, "test", prop=target, folder=str(save_folder + "/Fold {}".format(i)),
                         model="SVR", fold=i)

        with open(str(save_folder + "/Fold {}/svr_test_predictions_{}.txt".format(i, i)), "w") as f:
            f.write(str("Molecule \t Real Value \t Prediction \t Absolute Error \n"))
            for m, v, p, e in zip(test_molecules, y_test, y_svr, svr_error):
                f.write(str(m) + '\t' + str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(e, 4)) + '\n')
            f.close()

        return test_mean_absolute_error, test_root_mean_squared_error, \
               svr_mean_absolute_error, svr_root_mean_squared_error
    else:
        return test_mean_absolute_error, test_root_mean_squared_error
