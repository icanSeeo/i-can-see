import torch
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def extract_feature_maps(model, test_image):
    desired_layer = 'layer4' 
    
    # 특징 맵 추출
    feature_maps = None
    hooks = []
    
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output
    
    for name, layer in model.named_modules():
        if name == desired_layer:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    with torch.no_grad():
        model(test_image)
    
    for hook in hooks:
        hook.remove()
    
    return feature_maps

def reduce_dimensions(all_feature_maps, method='pca', compress=False):
    print("[*] 1. reduce dimensions")
    reduced_all_features = []
    
    for feature_maps in all_feature_maps:
        combined_feature_maps = torch.cat(tuple(feature_maps), dim=0)
        flattened_feature_maps = combined_feature_maps.view(
            combined_feature_maps.size(0), -1).cpu().numpy()
        
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=10)
        else:
            raise ValueError('Wrong method name')

        reduced_features = torch.from_numpy(reducer.fit_transform(flattened_feature_maps))
        mean_feature = torch.mean(reduced_features, dim=0)
        
        # print('flattened :', flattened_feature_maps.shape)
        # print('combined :', combined_feature_maps.shape)
        # print('flattened T:', flattened_feature_maps.T.shape)
        # print('reduced featuremap shape ', reduced_features.shape)
        # print('mean featuremap shape ', mean_feature.shape)
        # print()
        
        if compress:
            reduced_all_features.append(mean_feature.unsqueeze(0))
        else:
            reduced_all_features.append(reduced_features)
        
        
    return reduced_all_features