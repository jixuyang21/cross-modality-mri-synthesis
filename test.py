import os

from data import create_dataloader
from options import Options
from models import create_model
from utils import make_dirs, get_logger, MetricsEvaluatorRecoder, make_epoch_dirs


opt = Options().parse()

make_dirs(opt.logs_path, os.path.join(opt.results_path, 'test_results'))
make_epoch_dirs(os.path.join(opt.results_path, 'test_results'))

test_loader, test_length = create_dataloader(opt, mode='test')

model = create_model(opt)
model.load(opt.model_epoch)

eval_recorder = MetricsEvaluatorRecoder(opt)
logger = get_logger(opt, mode='test')


eval_recorder.evaluate_model(opt, model, test_loader, is_val=False)