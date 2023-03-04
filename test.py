import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import paddle
import yaml
from easydict import EasyDict as edict
import datasets


def test():
    net.eval()
    for batch_idx, (rgb, lidar, _, idx, ori_size) in enumerate(testloader):
        with paddle.no_grad():
            if config.tta:
                rgbf = paddle.flip(rgb, [-1])
                lidarf = paddle.flip(lidar, [-1])
                rgbs = paddle.concat([rgb, rgbf], 0)
                lidars = paddle.concat([lidar, lidarf], 0)
                depth_preds = net(rgbs, lidars)
                depth_pred, depth_predf = depth_preds.split(2)
                depth_predf = paddle.flip(depth_predf, [-1])
                depth_pred = (depth_pred + depth_predf) / 2.0
            else:
                depth_pred = net(rgb, lidar)
            depth_pred[depth_pred < 0] = 0
        depth_pred = depth_pred.squeeze(1).numpy()
        idx = idx.squeeze(1).numpy()
        ori_size = ori_size.numpy()
        name = [testset.names[i] for i in idx]
        save_result(config, depth_pred, name, ori_size)


if __name__ == "__main__":
    # config_name = 'GN.yaml'
    config_name = "GNS.yaml"
    with open(os.path.join("configs", config_name), "r") as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    from utils import *

    transform = init_aug(config.test_aug_configs)
    key, params = config.data_config.popitem()
    dataset = getattr(datasets, key)
    testset = dataset(
        **params, mode="selval", transform=transform, return_idx=True, return_size=True
    )
    testloader = paddle.io.DataLoader(
        testset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    print("num_test = {}".format(len(testset)))
    net = init_net(config)
    paddle.device.cuda.empty_cache()
    net, _ = resume_state(config, net, name="best")
    test()
