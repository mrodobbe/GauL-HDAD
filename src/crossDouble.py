import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import Model
import pickle
from src.makeModel import model_builder
from src.makeMolecule import denormalize
from sklearn.svm import SVR
from src.plots import performance_plot
from joblib import wrap_non_picklable_objects
from sklearn.model_selection import KFold


@wrap_non_picklable_objects
def run_cv(all_molecules, all_heavy, x, y, loop, i, save_folder, target):
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
    n_folds = 9  # TODO: Make argument
    kf = KFold(n_folds, shuffle=True, random_state=12041997)

    rmse_ann = []
    rmse_svr = []
    mae_ann = []
    mae_svr = []
    models = []
    turn = 0

    for train2, val2 in kf.split(x_train_all):
        turn += 1
        x_train2 = x_train_all[train2]
        y_train2 = y_train_all[train2]
        x_val2 = x_train_all[val2]
        y_val2 = y_train_all[val2]

        if len(y.shape) == 2:
            output_layer_size = y.shape[1]
        else:
            output_layer_size = 1

        model = model_builder(x_train_all, output_layer_size, target)
        model.summary()

        print('{} training molecules, {} validation molecules'.format(len(x_train2),
                                                                      len(x_val2)))

        es = EarlyStopping(patience=150, restore_best_weights=True, min_delta=0.01, mode='min')

        class LossHistory(Callback):
            def on_epoch_end(self, batch, logs={}):
                print('{:.3f}\t\t-\t{:.3f}'.format(logs.get('val_mean_absolute_error'), np.sqrt(logs.get('val_loss'))))

        lh = LossHistory()

        print('Validation MAE\t-\tValidation MSE')
        history = model.fit(x_train2, y_train2, epochs=1000,
                            validation_data=(x_val2, y_val2),
                            batch_size=8, callbacks=[es, lh], verbose=0)

        if target != "cp":
            validation_predictions = model.predict(x_val2).reshape(-1)
        else:
            validation_predictions = np.asarray(model.predict(x_val2)).astype(np.float)
        if target != "cp":
            test_predictions = model.predict(x_test).reshape(-1)
        else:
            test_predictions = np.asarray(model.predict(x_test)).astype(np.float)

        intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('layer_3').output)
        training_intermediates = np.asarray(intermediate_layer(x_train_all)).astype(np.float)
        test_intermediates = np.asarray(intermediate_layer(x_test)).astype(np.float)

        models.append(model)

        if target != "cp":
            krr = SVR(kernel="rbf", gamma='scale', C=2.5e3)  # This is the support vector machine.
            # Try to find an algorithm that optimizes gamma and C. You can also add an epsilon factor
            krr.fit(training_intermediates, y_train_all)  # Execute regression

            y_svr = krr.predict(test_intermediates)  # Prediction for the test set (unseen data)
            if target == "s":
                y_svr = denormalize(y_svr, heavy_test, target, coefficient=1.5)
                y_test_svr = denormalize(y_test, heavy_test, target, coefficient=1.5)
            else:
                y_test_svr = y_test
            svr_error = np.abs(y_svr - y_test_svr)
            svr_mean_absolute_error = np.average(svr_error)
            mae_svr.append(svr_mean_absolute_error)
            svr_root_mean_squared_error = np.sqrt(np.average(svr_error ** 2))
            rmse_svr.append(svr_root_mean_squared_error)

            print('Test performance statistics for SVR:')
            print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(svr_mean_absolute_error))
            print('Root mean squared error:\t{:.2f} kJ/mol'.format(svr_root_mean_squared_error))

        if target != "h":
            test_predictions = denormalize(test_predictions, heavy_test, target, coefficient=1.5)
            y_test_normalized = denormalize(y_test, heavy_test, target, coefficient=1.5)
        else:
            y_test_normalized = y_test

        test_error = np.abs(test_predictions - y_test_normalized)
        test_mean_absolute_error = np.average(test_error)
        mae_ann.append(test_mean_absolute_error)
        test_root_mean_squared_error = np.sqrt(np.average(test_error ** 2))
        rmse_ann.append(test_root_mean_squared_error)

        print('Test performance statistics for ANN:')
        print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(test_mean_absolute_error))
        print('Root mean squared error:\t{:.2f} kJ/mol'.format(test_root_mean_squared_error))

        with open(str(save_folder + "/test_results_fold_{}_{}.txt".format(i, turn)), "w") as f:
            f.write('ANN Test performance statistics:\n')
            f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(test_mean_absolute_error))
            f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(test_root_mean_squared_error))
            if target != "cp":
                f.write('SVR Test performance statistics:\n')
                f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(svr_mean_absolute_error))
                f.write('Root mean squared error:\t{:.2f} kJ/mol\n'.format(svr_root_mean_squared_error))
            f.close()

    ann_index = np.argmin(rmse_ann)

    if target != "cp":
        svr_index = np.argmin(rmse_svr)

    test_mean_absolute_error = mae_ann[ann_index]
    test_root_mean_squared_error = rmse_ann[ann_index]
    if target != "cp":
        svr_mean_absolute_error = mae_svr[svr_index]
        svr_root_mean_squared_error = rmse_svr[svr_index]
    best_model = models[ann_index]
    best_model.save(str(save_folder + "/Fold {}".format(i)))

    print('Test performance statistics for ANN:')
    print('Mean absolute error:\t\t{:.2f} kJ/mol'.format(test_mean_absolute_error))
    print('Root mean squared error:\t{:.2f} kJ/mol'.format(test_root_mean_squared_error))

    if target != "cp":
        ensemble = np.array([])
        for model in models:
            test_predicted = model.predict(x_test).reshape(-1)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            if target == "s":
                test_predicted = denormalize(test_predicted, heavy_test, target, coefficient=1.5)
            if len(ensemble) == 0:
                ensemble = test_predicted
            else:
                ensemble = np.vstack((ensemble, test_predicted))
        ensemble_prediction = np.mean(ensemble, axis=0)
        ensemble_sd = np.std(ensemble, axis=0)
        if target == "s":
            y_test_nor = denormalize(y_test, heavy_test, target, coefficient=1.5)
        else:
            y_test_nor = y_test
        ensemble_error = np.abs(ensemble_prediction - y_test_nor)
        ensemble_mae = np.average(ensemble_error)
        ensemble_rmse = np.sqrt(np.average(ensemble_error ** 2))
    else:
        ensemble = np.array([])
        for model in models:
            test_predicted = model.predict(x_test)
            test_predicted = np.asarray(test_predicted).astype(np.float)
            prediction_shape = test_predicted.shape
            test_predicted = denormalize(test_predicted, heavy_test, target, coefficient=1.5)
            if len(ensemble) == 0:
                ensemble = test_predicted.flatten()
            else:
                ensemble = np.vstack((ensemble, test_predicted.flatten()))
        ensemble_prediction = np.average(ensemble, axis=0)
        ensemble_prediction = np.reshape(ensemble_prediction, prediction_shape)
        ensemble_sd = np.std(ensemble, axis=0)
        ensemble_sd = np.reshape(ensemble_sd, prediction_shape)
        y_test_nor = denormalize(y_test, heavy_test, target, coefficient=1.5)
        ensemble_error = np.abs(ensemble_prediction - y_test_nor)
        ensemble_mae = np.average(ensemble_error)
        ensemble_rmse = np.sqrt(np.average(ensemble_error ** 2))

    with open(str(save_folder + "/Fold {}/test_results_fold_{}.txt".format(i, i)), "w") as f:
        f.write('ANN Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(test_mean_absolute_error))
        f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(test_root_mean_squared_error))
        if target != "cp":
            f.write('SVR Test performance statistics:\n')
            f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(svr_mean_absolute_error))
            f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(svr_root_mean_squared_error))
        f.write('ANN Ensemble Test performance statistics:\n')
        f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(ensemble_mae))
        f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(ensemble_rmse))
        f.close()

    if target != "cp":
        best_predictions = best_model.predict(x_test).reshape(-1)
    else:
        best_predictions = np.asarray(best_model.predict(x_test)).astype(np.float)

    if target != "h":
        best_predictions = denormalize(best_predictions, heavy_test, target, coefficient=1.5)
        y_best_normalized = denormalize(y_test, heavy_test, target, coefficient=1.5)
    else:
        y_best_normalized = y_test

    best_error = np.abs(best_predictions - y_best_normalized)

    with open(str(save_folder + "/Fold {}/test_predictions_{}.txt".format(i, i)), "w") as f:
        f.write(str("Molecule \t Real Value \t Prediction \t Absolute Error \n"))
        for m, v, p, e in zip(test_molecules, y_test, best_predictions, best_error):
            if target == "cp":
                f.write(str(m) + '\t' + str(round(v[0], 4)) + '\t' + str(round(p[0], 4)) + '\t' +
                        str(round(e[0], 4)) + '\n')
            else:
                f.write(str(m) + '\t' + str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(e, 4)) + '\n')
        f.close()

    with open(str(save_folder + "/Fold {}/test_representations_fold_{}.pickle".format(i, i)), "wb") as f:
        pickle.dump(x_test, f)
    print("Dumped the test molecules!")

    with open(str(save_folder + "/Fold {}/test_outputs_fold_{}.pickle".format(i, i)), "wb") as f:
        pickle.dump(y_test, f)
    print("Dumped the test outputs!")

    with open(str(save_folder + "/Fold {}/test_ensemble_predictions_{}.txt".format(i, i)), "w") as f:
        f.write(str("Molecule \t Real Value \t Prediction \t Deviation \t Error \n"))
        for m, v, p, s, e in zip(test_molecules, y_test_nor, ensemble_prediction, ensemble_sd, ensemble_error):
            if target == "cp":
                f.write(str(m) + '\t' + str(round(v[0], 4)) + '\t' + str(round(p[0], 4)) + '\t' +
                        str(round(s[0], 4)) + '\t' + str(round(e[0], 4)) + '\n')
            else:
                f.write(str(m) + '\t' + str(round(v, 4)) + '\t' + str(round(p, 4)) + '\t' + str(round(s, 4)) +
                        '\t' + str(round(e, 4)) + '\n')
        f.close()

    # performance_plot(y_test, test_predictions, "test", prop=target, folder=str(save_folder + "/Fold {}".format(i)),
    #                  model="ANN", fold=i)
    if target != "cp":
        return test_mean_absolute_error, test_root_mean_squared_error, ensemble_mae, ensemble_rmse, \
               svr_mean_absolute_error, svr_root_mean_squared_error,
    else:
        return test_mean_absolute_error, test_root_mean_squared_error, ensemble_mae, ensemble_rmse
