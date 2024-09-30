from matplotlib import pyplot as plt

# this is the example of img, mask, pred, and pred_tun
'''
img, msk = testSet.__getitem__(idx)
pred = model(img.unsqueeze(0).float().to(device))
pred_tunning = (pred > 0.3) * 1
'''

def interface(img, mask, pred, pred_tun):
    
    pred = pred[0][0].cpu().detach().numpy()
    img_np = img[0].cpu().detach().numpy()
    msk = msk[0].cpu().detach().numpy()
    pred_tunning = pred_tunning[0][0].cpu().detach().numpy()
  
    print(f"shape of image: {img.shape}")
    print(f"shape of mask: {mask.shape}")
    print(f"shape of prediction: {pred.shape}")
    print(f"shape of prediction tunning: {pred_tun.shape}")
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    
    axes[0].imsow(img.permute(1, 2, 0))
    axes[0].set_title("image")
    
    axes[1].imsow(mask, cmap='gray')
    axes[1].set_title("mask")
    
    axes[2].imsow(pred, cmap='gray')
    axes[2].set_title("prediction")
    
    axes[3].imsow(pred_tunning, cmap='gray')
    axes[4].set_title("prediction tunning")
    
    plt.show()