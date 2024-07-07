import os
import sys

from torch.utils.tensorboard import SummaryWriter

from data import create_dataloader
from models import create_model
from options import Options
from utils import make_dirs, get_logger, get_train_batch_loss, MetricsEvaluatorRecoder, make_epoch_dirs


opt = Options().parse()

make_dirs(opt.ckpts_path, opt.logs_path, opt.results_path)
make_epoch_dirs(opt.results_path)

train_loader, train_length = create_dataloader(opt, mode='train')
val_loader, val_length = create_dataloader(opt, mode='val')

model = create_model(opt)

eval_recorder = MetricsEvaluatorRecoder(opt)

if opt.check_loss:
    writer = SummaryWriter(os.path.join(opt.logs_path, 'train_loss_curve'))
logger = get_logger(opt, mode='train')
logger.info(str(opt) + '\n' + '-' * 10)
logger_val = get_logger(opt, mode='val')

for epoch in range(opt.num_epochs):
    train_loss = 0
    model.train_mode()
    for i, batch in enumerate(train_loader):
        real_A = batch[opt.mod_in]
        real_B = batch[opt.mod_out]

        _, loss_G = model.forward(real_A, real_B)
        train_loss += loss_G.item()
        loss_D = model.optimize()

        info = get_train_batch_loss(opt, epoch, i, train_loader, (loss_G, loss_D))
        logger.info(info)
        sys.stdout.write('\r' + info)
    
    model.update_lr()
    train_loss /= train_length
    val_loss = eval_recorder.evaluate_model(opt, model, val_loader, logger=logger_val, epoch=epoch+1, dataset_length=val_length) / val_length
    eval_recorder.draw_metrics_curve(writer, epoch + 1)
    
    loss_info = "[epoch %d/%d]\n[train loss %f] [val loss %f]" % \
            (epoch + 1, opt.num_epochs, train_loss, val_loss)
    logger.info(loss_info + '\n' + '-' * 10 + '\n')
    if opt.check_loss:
        writer.add_scalars('epoch loss', {'train_loss':train_loss, 'val_loss':val_loss}, epoch + 1)

    if (epoch + 1) <= opt.save_end and (epoch + 1 - opt.save_start) % opt.save_interval == 0:
        model.save(epoch + 1)

if opt.check_loss:
    writer.close()