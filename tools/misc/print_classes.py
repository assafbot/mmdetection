import argparse

from mmengine.registry import init_default_scope
from mmengine import Config, DATASETS


def main(cfg):
    cfg = Config.fromfile(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = cfg.train_dataloader.dataset
    dataset = DATASETS.build(dataset)
    print(dataset.metainfo['classes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('cfg', help='Dataset config file')
    args = parser.parse_args()
    main(args.cfg)
