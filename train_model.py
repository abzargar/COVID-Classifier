from utils import*
# =============================================================================
# source_dir
# =============================================================================
source_dir='./'
# =============================================================================
# load mat files
# =============================================================================
covid_features=sio.loadmat(os.path.join(source_dir,'covid.mat')) 
covid_features=covid_features['covid'] 

normal_features=sio.loadmat(os.path.join(source_dir,'normal.mat')) 
normal_features=normal_features['normal']  

pneumonia_features=sio.loadmat(os.path.join(source_dir,'pneumonia.mat')) 
pneumonia_features=pneumonia_features['pneumonia']    
# =============================================================================
# devide feature pool into inputs and target labels
# =============================================================================
X=np.concatenate((covid_features[:,:-1],normal_features[:,:-1],pneumonia_features[:,:-1]), axis=0)#inputs
y=np.concatenate((covid_features[:,-1],normal_features[:,-1],pneumonia_features[:,-1]), axis=0)#target labels
# =============================================================================
# normalization
# =============================================================================
min_max_scaler=MinMaxScaler()
X = min_max_scaler.fit_transform(X) 
# =============================================================================
# feature reduction (K-PCA)
# =============================================================================
transformer = KernelPCA(n_components=64, kernel='linear')
X = transformer.fit_transform(X)
# =============================================================================
# devide data into test,train, and validation sets
# =============================================================================
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)
# =============================================================================
# build model
# =============================================================================
model = build_model(feature_size=X.shape[-1], n_classes=y.shape[-1])
show_model(model)
# =============================================================================
# train model
# =============================================================================
opt = tf.keras.optimizers.Adam(lr=0.001)
criterion = tf.keras.losses.categorical_crossentropy
model.compile(optimizer=opt, loss=criterion,metrics=[keras.metrics.categorical_accuracy])
model_early_stopping=EarlyStopping(monitor='val_loss', min_delta=.005, patience=5, verbose=1)# early stopping settings
model.fit(X_train, y_train,batch_size = 2,epochs=100, validation_data = (X_val, y_val),callbacks=[model_early_stopping])

fig = plt.figure()
plt.plot(model.history.history['val_loss'], 'r', model.history.history['loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Loss Score')
plt.grid(1)
plt.savefig('training_loss.jpg',dpi=300)
# =============================================================================
# evaluate model
# =============================================================================
Y_Score=model.predict(X_test)
y_pred = np.argmax(Y_Score, axis=1)
cm=confusion_matrix(np.argmax(y_test, axis=1),y_pred)
print(cm)

fig = plt.figure()
plot_confusion_matrix(cm,classes=['COVID-19','Normal','Pneumonia'])
plt.savefig('conf_matrix.jpg',dpi=300)

test_loss=model.evaluate(X_test,y_test,verbose=1)#evaluate model
print(test_loss)#print test loss and metrics information

# =============================================================================
# ROC curve where positive label is COVID-19 
# =============================================================================
pos_label=0
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), Y_Score[:,pos_label], pos_label=pos_label)
roc_auc = auc(fpr, tpr)# calculate auc value
fig = plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)\npos_label=COVID-19' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.grid(1)
plt.legend(loc="lower right")
plt.savefig('roc_covid.jpg',dpi=300)

# =============================================================================
# ROC curve where positive label is normal
# =============================================================================
pos_label=1
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), Y_Score[:,pos_label], pos_label=pos_label)
roc_auc = auc(fpr, tpr)# calculate auc value
fig = plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)\npos_label=Normal' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.grid(1)
plt.legend(loc="lower right")
plt.savefig('roc_normal.jpg',dpi=300)

# =============================================================================
# ROC curve where positive label is COVID-19
# =============================================================================
pos_label=2
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), Y_Score[:,pos_label], pos_label=pos_label)
roc_auc = auc(fpr, tpr)# calculate auc value
fig = plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)\npos_label=Pneumonia' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.grid(1)
plt.legend(loc="lower right")
plt.savefig('roc_pneumonia.jpg',dpi=300)






