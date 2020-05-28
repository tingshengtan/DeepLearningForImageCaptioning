import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

mainDirectory = "/Users/tingsheng/OneDrive/VersionControl/EdinburghMlpGroupProject/EdinburghMlpClusterTutorial/"

resnetLstm64ResultsPath    = os.path.join(mainDirectory, 'image_caption_resnet_lstm_glove_64_by_64_exp/result_outputs/summary.csv')
resnetTpgn64ResultsPath    = os.path.join(mainDirectory, 'image_caption_resnet_tpgn_glove_64_by_64_exp/result_outputs/summary.csv')
densenetLstm64ResultsPath  = os.path.join(mainDirectory, 'image_caption_densenet_lstm_glove_64_by_64_exp/result_outputs/summary.csv')
densenetTpgn64ResultsPath  = os.path.join(mainDirectory, 'image_caption_densenet_tpgn_glove_64_by_64_exp/result_outputs/summary.csv')
resnetLstm256ResultsPath   = os.path.join(mainDirectory, 'image_caption_resnet_lstm_glove_256_by_256_exp/result_outputs/summary.csv')
resnetTpgn256ResultsPath   = os.path.join(mainDirectory, 'image_caption_resnet_tpgn_glove_256_by_256_exp/result_outputs/summary.csv')
densenetLstm256ResultsPath = os.path.join(mainDirectory, 'image_caption_densenet_lstm_glove_256_by_256_exp/result_outputs/summary.csv')
densenetTpgn256ResultsPath = os.path.join(mainDirectory, 'image_caption_densenet_tpgn_glove_256_by_256_exp/result_outputs/summary.csv')

outputPath = os.path.join(mainDirectory, 'experiment_results/')

resnetLstm64Results    = pd.read_csv(resnetLstm64ResultsPath, delimiter=',')
resnetTpgn64Results    = pd.read_csv(resnetTpgn64ResultsPath, delimiter=',')
densenetLstm64Results  = pd.read_csv(densenetLstm64ResultsPath, delimiter=',')
densenetTpgn64Results  = pd.read_csv(densenetTpgn64ResultsPath, delimiter=',')
resnetLstm256Results   = pd.read_csv(resnetLstm256ResultsPath, delimiter=',')
resnetTpgn256Results   = pd.read_csv(resnetTpgn256ResultsPath, delimiter=',')
densenetLstm256Results = pd.read_csv(densenetLstm256ResultsPath, delimiter=',')
densenetTpgn256Results = pd.read_csv(densenetTpgn256ResultsPath, delimiter=',')

# print(resnetLstm64Results.head())

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm64Results['train_acc'].values[:20]
y2 = resnetTpgn64Results['train_acc'].values[:20]
y3 = densenetLstm64Results['train_acc'].values[:20]
y4 = densenetTpgn64Results['train_acc'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.xticks(x)
plt.title('Model Training Accuracy (64x64 resolution images)')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'TrainingAccuracy64.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm64Results['train_loss'].values[:20]
y2 = resnetTpgn64Results['train_loss'].values[:20]
y3 = densenetLstm64Results['train_loss'].values[:20]
y4 = densenetTpgn64Results['train_loss'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.xticks(x)
plt.title('Model Training Loss (64x64 resolution images)')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'TrainingLoss64.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm64Results['val_acc'].values[:20]
y2 = resnetTpgn64Results['val_acc'].values[:20]
y3 = densenetLstm64Results['val_acc'].values[:20]
y4 = densenetTpgn64Results['val_acc'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Validation Accuracy (64x64 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'ValidationAccuracy64.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm64Results['val_loss'].values[:20]
y2 = resnetTpgn64Results['val_loss'].values[:20]
y3 = densenetLstm64Results['val_loss'].values[:20]
y4 = densenetTpgn64Results['val_loss'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Validation Loss (64x64 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'ValidationLoss64.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm64Results['bleu4'].values[:20]
y2 = resnetTpgn64Results['bleu4'].values[:20]
y3 = densenetLstm64Results['bleu4'].values[:20]
y4 = densenetTpgn64Results['bleu4'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Validation BLEU-4 (64x64 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('BLEU-4')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'Blue4Score64.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm256Results['train_acc'].values[:20]
y2 = resnetTpgn256Results['train_acc'].values[:20]
y3 = densenetLstm256Results['train_acc'].values[:20]
y4 = densenetTpgn256Results['train_acc'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Training Accuracy (256x256 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'TrainingAccuracy256.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm256Results['train_loss'].values[:20]
y2 = resnetTpgn256Results['train_loss'].values[:20]
y3 = densenetLstm256Results['train_loss'].values[:20]
y4 = densenetTpgn256Results['train_loss'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Training Loss (256x256 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'TrainingLoss256.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm256Results['val_acc'].values[:20]
y2 = resnetTpgn256Results['val_acc'].values[:20]
y3 = densenetLstm256Results['val_acc'].values[:20]
y4 = densenetTpgn256Results['val_acc'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Validation Accuracy (256x256 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'ValidationAccuracy256.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm256Results['val_loss'].values[:20]
y2 = resnetTpgn256Results['val_loss'].values[:20]
y3 = densenetLstm256Results['val_loss'].values[:20]
y4 = densenetTpgn256Results['val_loss'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Validation Loss (256x256 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'ValidationLoss256.png'), dpi=400)

fig = plt.figure()
x  = np.arange(0, 20)
y1 = resnetLstm256Results['bleu4'].values[:20]
y2 = resnetTpgn256Results['bleu4'].values[:20]
y3 = densenetLstm256Results['bleu4'].values[:20]
y4 = densenetTpgn256Results['bleu4'].values[:20]
plt.plot(x, y1, label='ResNet + LSTM')
plt.plot(x, y2, label='ResNet + TPGN')
plt.plot(x, y3, label='DenseNet + LSTM')
plt.plot(x, y4, label='DenseNet + TPGN')
plt.title('Model Validation BLEU-4 (256x256 resolution images)')
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('BLEU-4')
plt.legend(loc='best')
fig.tight_layout()
plt.savefig(os.path.join(outputPath, 'Blue4Score256.png'), dpi=400)


