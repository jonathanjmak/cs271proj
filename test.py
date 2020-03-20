import cv2
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions
#%%

if __name__ == '__main__':
    
    args = parse_args()
   
    if args.model not in model_dict:
        print ("Model unavailable.")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = args.load
    if not os.path.exists(filename):
        print("incorrect model path")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath = '/home/jjonathanmak/cs271proj/Semantic_Segmentation_Dataset',\
                                 split = 'test',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=2)
    
    
    os.makedirs('test/labels/',exist_ok=True)
    os.makedirs('test/output/',exist_ok=True)
    os.makedirs('test/mask/',exist_ok=True)
    
#     # for keratitis debug testing:
#     filepath = '/home/jjonathanmak/cs271proj/images_preprocessed/test/images'
#     with torch.no_grad():
#         print("Keratitis Testing")
#         for im in os.listdir(filepath):
#             data = cv2.imread(filepath+'/'+im)      
#             output = model(torch.from_numpy(data))
#             predict = get_predictions(output)
#             for j in range (len(index)):       
#                 np.save('test/labels/{}.npy'.format(index[j]),predict[j].cpu().numpy())
#                 try:
#                     plt.imsave('test/output/{}.jpg'.format(index[j]),255*labels[j].cpu().numpy())
#                 except:
#                     pass

#                 pred_img = predict[j].cpu().numpy()/3.0
#                 inp = img[j].squeeze() * 0.5 + 0.5
#                 img_orig = np.clip(inp,0,1)
#                 img_orig = np.array(img_orig)
#                 combine = np.hstack([img_orig,pred_img])
#                 plt.imsave('test/mask/{}.jpg'.format(index[j]),combine)

# all test
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
            img,labels,index,x,y= batchdata
            data = img.to(device)       
            output = model(data)            
            predict = get_predictions(output)
            for j in range (len(index)):       
                np.save('test/labels/{}.npy'.format(index[j]),predict[j].cpu().numpy())
                try:
                    plt.imsave('test/output/{}.jpg'.format(index[j]),255*labels[j].cpu().numpy())
                except:
                    pass
                
                pred_img = predict[j].cpu().numpy()/3.0
                inp = img[j].squeeze() * 0.5 + 0.5
                img_orig = np.clip(inp,0,1)
                img_orig = np.array(img_orig)
                combine = np.hstack([img_orig,pred_img])
                plt.imsave('test/mask/{}.jpg'.format(index[j]),combine)

    os.rename('test',args.save)
