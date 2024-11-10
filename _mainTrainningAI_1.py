import cv2
import numpy as np
import pandas as pd
import os
import random
from datetime import datetime
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from _findObject import find_object
from sklearn import preprocessing

def trainning_1(dataSize, myEpochs, model_number, learningRate = 0.001, myDropOut = False):
    project_directory = os.path.join(os.getcwd(), 'dataFolder_240805')
    folder_path = os.path.join(project_directory, 'stereoVisionPictures2')
    csv_file = os.path.join(folder_path, 'data.txt')
    
    items = os.listdir(project_directory)
    # Iterate through the items and find the directory name starting with "model_"
    """model_number = 1
    for item in items:
        if item.startswith("cnnModel_"):
            if int(item.split('_')[1]) > model_number:
                model_number = int(item.split('_')[1])"""

    model_path = os.path.join(project_directory, 'cnnModel_' + model_number)
    os.mkdir(model_path)
    print(model_path)

    # Load images and labels
    dataTXT = pd.read_csv(csv_file)

    output_data = []
    trainning_points = []
    count = 0

    for i in range(2, dataSize+4):
        try:
            capL = os.path.join(folder_path, 'L_' + str(i) + '.jpg')
            capR = os.path.join(folder_path, 'R_' + str(i) + '.jpg')

            frameL, centerL, boxL, frameR, centerR, boxR, img_dilationL, img_dilationR, detectionCounts = find_object(capL, capR, 
                                                                                        tresholdValueL = 50, tresholdValueR = 50, 
                                                                                        minArea = 10000, maxArea = 500000, 
                                                                                        histogram = False, drawContours = True)
            
            if detectionCounts > 2:
                frame = cv2.hconcat([cv2.resize(frameL, (640, 420)), cv2.resize(frameR, (640, 420))])
                gframe = cv2.hconcat([cv2.resize(img_dilationL, (640, 420)), cv2.resize(img_dilationR, (640, 420))])
                cv2.imshow('Webcams Feed', frame)
                cv2.imshow('Webcams Feed B&W', gframe)
                cv2.waitKey(3)
                input("Two Detections, press Enter to continue...")
                
            x = dataTXT.loc[i-2, "X"]
            y = dataTXT.loc[i-2, "Y"]
            z = dataTXT.loc[i-2, "Z"]
            u = dataTXT.loc[i-2, "U"]

            # Normalize data
            # Picture size: 720 x 1280 px
            #boxL[0][:] = boxL[0][:]/1280 
            #boxL[:][1] = boxL[:][1]/720

            #boxR[:][0] = boxR[:][0]/1280 
            #boxR[:][1] = boxR[:][1]/720

            #temp = np.concatenate((boxL, boxR), axis = 0).flatten()
            temp = [centerL[0]/1280, centerL[1]/720,  centerR[0]/1280, centerR[1]/720,
                    boxL[0][0]/1280, boxL[0][1]/720, boxL[1][0]/1280, boxL[1][1]/720, boxL[2][0]/1280, boxL[2][1]/720, boxL[3][0]/1280, boxL[3][1]/720, 
                    boxR[0][0]/1280, boxR[0][1]/720, boxR[1][0]/1280, boxR[1][1]/720, boxR[2][0]/1280, boxR[2][1]/720, boxR[3][0]/1280, boxR[3][1]/720]
            
            trainning_points.append(temp)
            output_data.append([x, y, z, u])
            print(capL)
            #print(i, dataTXT.loc[i, "ID"])
            count = count + 1
            #print(count)
        except:
            pass
    #print(trainning_points[0:5])
    #print(output_data[0:5])
    #trainning_points /= 1000.0
    #trainning_points = preprocessing.normalize([trainning_points])
    
    #trainning_points = tf.convert_to_tensor(trainning_points)

    # Definindo a arquitetura da rede neural
    if myDropOut:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),  # Camada de entrada
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta1
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta2
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta3
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta4
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta5
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta6
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta7
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta8
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta9
            tf.keras.layers.Dropout(0.2),  # Adicionando dropout
            tf.keras.layers.Dense(4)  # Camada de saída
        ])
    else:
        model = tf.keras.Sequential([
            #tf.keras.layers.Flatten(input_shape=(8, 2)),  # Camada de entrada
            tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),  # Camada de entrada
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta1
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta2
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta3
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta4
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta5
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta6
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta7
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta8
            tf.keras.layers.Dense(16, activation='relu'), # Camada oculta9
            tf.keras.layers.Dense(4)  # Camada de saída
            #,activation=keras.layers.LeakyReLU(alpha=0.01)
        ])
    
    # Embaralhar e dividir os dados
    #trainning_points = tf.convert_to_tensor(trainning_points, dtype = tf.float32)
    train_input, val_input, train_output, val_output = train_test_split(trainning_points, output_data, test_size = 0.10*(dataSize/(dataSize+4)), 
                                                                        train_size = 0.90*(dataSize/(dataSize+4)), shuffle=True, stratify=None) 

    #print(train_input[:10])
    #print(train_output[:10])

    # Compilando o modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)  # You can adjust the learning rate
    model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['accuracy','mae', 'mape'])   

    checkpoint_path = os.path.join(model_path, 'ckpt')

    # Create a callback that saves the model's weights
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_loss', # or val_accuracy if you have it.
                                                    save_best_only=True, # Default false. If you don't change the file name then the output will be overritten at each step and only the last model will be saved.
                                                    save_weights_only=True, # True => model.save_weights (weights and no structure, you need JSON file for structure), False => model.save (saves weights & structure)
                                                    verbose=0,
                                                    )
    
    #input(f'Number of data : {count}; Number of training data: {len(train_input)}; Number of validation data: {len(val_input)}.')
    #input('Press enter to continue...')

    start = datetime.now() 
    # Treinamento do modelo
    history = model.fit(np.array(train_input), np.array(train_output), epochs=myEpochs, 
                        validation_data=(np.array(val_input), np.array(val_output)), 
                        shuffle=True,
                        callbacks=[ckpt_callback])
    end = datetime.now()

    # Salvar o modelo treinado
    model.save(model_path)
    ##print('ckpnt')
    #model.save_weights(model_path)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(model_path + '/accuracy_plot.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_path + '/loss_plot.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([myEpochs-1000, myEpochs])
    plt.ylim([0, 4000])
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_path + '/loss_plot_detail.png')
    plt.clf()
    # summarize history for mae
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_path + '/mae_plot.png')
    plt.clf()
    # summarize history for mse
    plt.plot(history.history['mape'])
    plt.plot(history.history['val_mape'])
    plt.title('model mape')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_path + '/mape_plot.png')
    plt.clf()

    
    #model.summary()
    #print(f'Number of data : {count}; Number of training data: {len(train_input)}; Number of validation data: {len(val_input)}.')
    #print(train_output[5])
    #print(predictions)
    #print(((end - start).total_seconds())/3600)


    #os.path.join(project_directory, 'cnnModel_'+str(model_number))
    f = open(model_path + '/infos_' + model_number + '.txt', "w")
    #f.write(f'Model number: {model_number}\n')
    f.write(f'Trainning time : {((end - start).total_seconds())/3600}\n')
    for i in range(0, myEpochs, 1000):
        lossTemp = history.history['loss'][i] 
        valLossTemp = history.history['val_loss'][i] 
        f.write(f'{i};{lossTemp};{valLossTemp}\n')
    lossTemp = history.history['loss'][-1]
    valLossTemp = history.history['val_loss'][-1]
    f.write(f'{myEpochs};{lossTemp};{valLossTemp}\n')
    f.write(f'Number of data : {count}\nNumber of training data: {len(train_input)}\nNumber of validation data: {len(val_input)}\n')
    f.write(f'learning_rate : {learningRate}\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 3. Fazer previsões
    f.write(f'Test point : {train_output[8]}\n')
    input_point = np.array(train_input[8]).reshape(1, -1)  # Reshape to (1, 16)
    predictions = model.predict(input_point)
    f.write(f'Reponse : {predictions}\n')
    
    f.write(f'Test point : {train_output[10]}\n')
    input_point = np.array(train_input[10]).reshape(1, -1)  # Reshape to (1, 16)
    predictions = model.predict(input_point)
    f.write(f'Reponse : {predictions}\n')
    
    f.write(f'Test point : {train_output[30]}\n')
    input_point = np.array(train_input[30]).reshape(1, -1)  # Reshape to (1, 16)
    predictions = model.predict(input_point)
    f.write(f'Reponse : {predictions}\n')
    
    f.close()

if __name__ == "__main__":
    #trainning_1(dataSize = 2800, myEpochs = 5000, model_number = 200, learningRate = 0.001, myDropOut = False)
    #trainning_1(dataSize = 5600, myEpochs = 5000, model_number = 201, learningRate = 0.001, myDropOut = False)
    #trainning_1(dataSize = 2800, myEpochs = 5000, model_number = 202, learningRate = 0.001, myDropOut = True)
    #trainning_1(dataSize = 5, myEpochs = 10, model_number = 208, learningRate = 0.001, myDropOut = False)
    #trainning_1(dataSize = 3800, myEpochs = 5000, model_number = 556, learningRate = 0.001, myDropOut = False)
    trainning_1(dataSize = 4506, myEpochs = 50000, model_number = '50k', learningRate = 0.001, myDropOut = False)
    trainning_1(dataSize = 4506, myEpochs = 50000, model_number = '50k_DO', learningRate = 0.001, myDropOut = True)