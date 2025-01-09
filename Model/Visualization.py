import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from get_dataset import *
import sklearn.metrics as sm
from sklearn import manifold
import numpy as np

def standardization(data):
    data = data
    mu = np.mean(data, axis=0)
    # print("mu:",mu,mu.shape)
    sigma = np.std(data, axis=0)
    # print("sigma:",sigma,sigma.shape)
    return (data - mu) / sigma

def scatter(features, targets, subtitle = None, n_classes = 10):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)

def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure()
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)


def TestDataset_prepared(signal_file, label_file,min_value,max_value):
    X_test, Y_test = np.load(signal_file).transpose(0, 2, 1),np.load(label_file)

    
    X_test = (X_test - min_value) / (max_value - min_value)

    return X_test, Y_test

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            #target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #target = target.to(device)
            # output = model(data)
            output = model.encoder(data)
            # print(output.shape)
            feature_map[len(feature_map):len(output[0])-1] = output.tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def main(num_classes,sample):
    data_path = f'Dataset_4800/X_train_{num_classes}Class.npy'
    # label_path = f'Dataset_4800/Y_train_{num_classes}Class.npy'
    data = np.load(data_path)
    min_value = np.min(data)
    max_value = np.max(data)
    # max_value = np.maximum(np.max(train_X), np.max(valid_X))
    
    data_path = f'Dataset_4800/X_test_{num_classes}Class.npy'
    label_path = f'Dataset_4800/Y_test_{num_classes}Class.npy'
    
    X_test, Y_test = TestDataset_prepared(data_path,label_path,min_value,max_value)
    print(X_test.shape)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # model = torch.load(f'model_idc_t=0.8/Finetuned_Model_{num_classes}c{sample}k_0.pkl')
    # model = torch.load(f'model_spt/Finetuned_Model_{num_classes}c{sample}k_0.pkl')
    model = torch.load(f'model_vat_bs32/Finetuned_Model_{num_classes}c{sample}k_0.pkl')
    # model = torch.load(f'model_idc_nocl/Finetuned_Model_{num_classes}c{sample}k_0.pkl')
    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)
    print(X_test_embedding_feature_map.shape,target.shape)
    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    print(eval_tsne_embeds.shape)
    scatter(eval_tsne_embeds, target.astype('int64'), f"{num_classes}c{sample}k", num_classes)
    # visualize_data(X_test_embedding_feature_map, target.astype('int64'), f"Visualization_t=0.8/{num_classes}c{sample}k", num_classes)
    visualize_data(X_test_embedding_feature_map, target.astype('int64'), f"Visualization_vat_bs32/{num_classes}c{sample}k", num_classes)
    print(sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean'))
    return sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean')
if __name__ == "__main__":
    Ks = [1, 5, 10,15,20] ##1, 5, 10,15,
    num_Ks = np.shape(Ks)[0]
    Ns = [10, 20, 30] #, 20, 30
    num_Ns = np.shape(Ns)[0]
    aa = []
    for n in range(num_Ns):
        for k in range(num_Ks):
            a = main(Ns[n],Ks[k])
            aa.append(a)
    print(aa)        