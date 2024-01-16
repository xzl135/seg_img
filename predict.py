import os
import torch
import torchvision.models.segmentation
import numpy as np
from matplotlib import pyplot as plt
import datasets_img

def save_prediction_as_image(image, pred, filename):
    fig, ax = plt.subplots()
    image = np.transpose(image, (1, 2, 0))
    ax.imshow(image)
    pred_binary = np.squeeze(pred > 0)
    pred_rgb = np.zeros((pred_binary.shape[0], pred_binary.shape[1], 3))
    pred_rgb[pred_binary] = [1, 0, 0]  # RGB for green
    ax.imshow(pred_rgb, alpha=0.5)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    # 参数设置
    batch_size = 10
    data_dir = 'rv_dataset'
    output = f'{data_dir}/output/test'
    num_workers = 4
    device = torch.device('cuda')
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=True)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1,kernel_size=model.classifier[-1].kernel_size)
    model.to(device)
    # 求均值方差
    mean, std = np.array([32.366024, 32.50149 , 32.80471 ], dtype=np.float32),np.array([49.646393, 49.66005 , 49.909855], dtype=np.float32)
    # 加载数据集
    checkpoint = torch.load(os.path.join(data_dir,'output' ,"best.pt"))
    model.load_state_dict(checkpoint['state_dict'])
    kwargs = {"mean": mean,
            "std": std
            }
        # Run on external test
    test_dataset = datasets_img.Echo(root=data_dir, split="ex_test", **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Run the model on the test data and save the results
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs_dict = model(inputs)
            outputs = outputs_dict['out']
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy() > 0.
            labels = labels.cpu().numpy()
            pred=outputs[0][0]
            save_prediction_as_image(inputs[0],  pred, f'{output}/{i}.png')
