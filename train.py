import logging

# 函数在执行时，要带上所有必要的参数进行调用。但是，有时参数可以在函数被调用之前提前获知。
#这种情况下，一个函数有一个或多个参数预先就能用上，以便函数能用更少的参数进行调用。
#偏函数是将所要承载的函数作为partial()函数的第一个参数，原函数的各个参数依次作为partial()函数后续的参数，除非使用关键字参数。
from functools import partial

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader

from adversarial_atrainer import GANFactory  # 生成GAN模型的
from dataset import PairedDataset
from metric_counter import MetricCounter    
from models.losses import get_loss    # 损失函数
from models.models import get_model   # 像是构建网络输入数据的？？
from models.networks import get_nets  # 构建生成器和判别器
from schedulers import LinearDecay, WarmRestart   

cv2.setNumThreads(0)


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']   # 0.001
        self.metric_counter = MetricCounter(config['experiment_desc'])   # fpn   记录实验过程的loss等信息
        self.warmup_epochs = config['warmup_num']  # 3 

    def train(self):
        self._init_params()  # 初始化网络等训练必要的东西
        for epoch in range(0, config['num_epochs']):  # 200
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.netG.state_dict()
            }, 'last_{}.h5'.format(self.config['experiment_desc']))
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()  # 重新建立字典
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:  #data:{a:(1,3,256,256) b:(1,3,256,256)}
            inputs, targets = self.model.get_input(data)
            outputs = self.netG(inputs)  #(1,3,256,256)
            loss_D = self._update_d(outputs, targets)  
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item(), loss_content.item(), loss_D)
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            outputs = self.netG(inputs)
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            self.metric_counter.add_losses(loss_G.item(), loss_content.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)  
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    # 构建损失函数，网络，优化器，调度器
    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, netD = get_nets(self.config['model'])
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)


if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f)

    batch_size = config.pop('batch_size')  # 1
    # 这里使用了python的偏函数，第一个参数是函数名，后续的参数是函数可以提前获知的参数
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)

    #获取训练，验证的配置信息
    datasets = map(config.pop, ('train', 'val'))  # map(function, iterable, ...)，
    # 初始化一个dataset对象，因为使用了静态方法
    datasets = map(PairedDataset.from_config, datasets)
    # 获得数据加载器
    train, val = map(get_dataloader, datasets)   # 这一步调用了dataset里的from_config方法
    trainer = Trainer(config, train=train, val=val)
    trainer.train()
    
