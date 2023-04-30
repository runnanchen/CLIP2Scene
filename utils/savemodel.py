import torch
import os

def save_checkpoint(self):
    trained_epoch = self.cur_epoch + 1
    ckpt_name = self.ckpt_dir / ('checkpoint_epoch_%d' % trained_epoch)
    checkpoint_state = {}
    checkpoint_state['epoch'] = trained_epoch
    checkpoint_state['it'] = self.it
    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(self.model.module.state_dict())
    else:
        model_state = model_state_to_cpu(self.model.state_dict())

    checkpoint_state['model_state'] = model_state
    checkpoint_state['optimizer_state'] = self.optimizer.state_dict()
    checkpoint_state['scaler'] = self.scaler.state_dict()
    checkpoint_state['lr_scheduler_state'] = self.lr_scheduler.state_dict()

    torch.save(checkpoint_state, f"{ckpt_name}.pth")

def resume(self, filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    self.logger.info(f"==> Loading parameters from checkpoint {filename}")
    checkpoint = torch.load(filename, map_location='cpu')
#        self.cur_epoch = checkpoint['epoch']
#        self.start_epoch = checkpoint['epoch']
#        self.it = checkpoint['it']
    self.model.load_params(checkpoint['model_state'], strict=True)
#        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#        self.scaler.load_state_dict(checkpoint['scaler'])
#        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
    self.logger.info('==> Done')
    return