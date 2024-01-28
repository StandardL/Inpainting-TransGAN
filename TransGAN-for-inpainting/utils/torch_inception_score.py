import sys

import numpy as np
import torch
from tqdm import tqdm
from torchvision.models.inception import inception_v3
from scipy.stats import entropy


def torch_get_inception_score(images, splits=10):
    assert type(images) == list
    assert type(images[0]) == np.ndarray
    assert len(images[0].shape) == 3
    assert np.max(images[0]) > 10
    assert np.min(images[0]) >= 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # InceptionV3模型提取特征
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    feature_layer = model.fc
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1], feature_layer).to(device)
    inp = []
    for img in images:
        img = img.astype(np.float32)
        inp.append(np.expand_dims(img, 0))
    bs = 128
    with torch.no_grad():
        predictions = []
        num_batches = int(np.ceil(float(len(inp))/float(bs)))
        for i in tqdm(range(num_batches), desc="Calculate inception score"):
            inp_tensor = inp[i*bs:min((i+1) * bs, len(inp))]
            inp_tensor = np.concatenate(inp_tensor, 0)
            inp_tensor = torch.from_numpy(inp_tensor).to(device)
            # with torch.no_grad():
            pred = torch.softmax(inp_tensor, dim=1)
            predictions.append(pred.cpu().numpy())
        predictions = np.concatenate(predictions, 0)
        scores = []
        for k in range(splits):
            part = predictions[(k * predictions.shape[0] // splits):((k + 1) * predictions.shape[0] // splits), :]
            py = np.mean(part, axis=0)
            s = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                s.append(entropy(pyx, py))
            scores.append(np.exp(np.mean(s)))
    return np.mean(scores), np.std(scores)
