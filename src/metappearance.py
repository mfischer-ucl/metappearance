import os
import copy
import time

import torch
import torch.nn as nn
import torchvision.io
import matplotlib.pyplot as plt

from utils.utils import zero_gradients
from torch.utils.tensorboard import SummaryWriter


class Metappearance(nn.Module):
    def __init__(self, cfg, model, lossfn, trainDistr, testDistr):
        super(Metappearance, self).__init__()

        self.model = model      # the trainable model whose weights will be optimized  
        self.loss_fn = lossfn   
        self.trainDistr, self.testDistr = trainDistr, testDistr

        self.cfg = cfg
        self.device = cfg.device

        # init inner- and outer-loop optimizers
        self.param_lr, self.inner_lr = {}, 0.0
        self.meta_optim = self.init_optimizers()

        # load weights and optim if needed
        if self.cfg.meta.load_checkpt is True:
            self.checkpt_path = os.path.join(self.cfg.general.basepath, self.cfg.meta.checkpt_path)
            self.load_checkpt(self.checkpt_path)

        # init rest: logging, writing, hparams, bookeeping, ...
        self.checkpt_savedir, self.writer = None, None
        self.checkpt_savedir = os.path.join(self.cfg.general.basepath,
                                            'hydra_logs/{}/{}/{}'.format(cfg.name, cfg.version, 'checkpts'))
        self.converged_savedir = self.checkpt_savedir.replace('checkpts', 'converged')
        self.log_savedir = self.checkpt_savedir.replace('checkpts', 'logs')
        self.output_savedir = self.checkpt_savedir.replace('checkpts', 'outputs')
        self.bestLoss = torch.tensor(1e9, device=self.device)
        self.init_bookkeeping()

    ############################################################
    # Init & Housekeeping Routines
    ############################################################

    def init_bookkeeping(self):
        if not os.path.exists(self.log_savedir): os.mkdir(self.log_savedir)
        if not os.path.exists(self.output_savedir): os.mkdir(self.output_savedir)
        if not os.path.exists(self.checkpt_savedir): os.mkdir(self.checkpt_savedir)
        if not os.path.exists(self.converged_savedir): os.mkdir(self.converged_savedir)
        self.writer = SummaryWriter(log_dir=self.log_savedir)

    def init_metaSGD(self):
        # meta-sgd: for each model parameter, add a learning rate
        for key, val in self.model.named_parameters():
            self.param_lr[key] = torch.nn.Parameter(self.cfg.meta.init_metasgd * torch.ones_like(val, requires_grad=True))

    def init_optimizers(self):
        # inner-loop optim:
        if self.cfg.meta.inner_optim == 'meta_sgd':
            self.init_metaSGD()
            optim_params = list(self.model.parameters()) + [self.param_lr[k] for k in self.param_lr.keys()]
        elif self.cfg.meta.inner_optim == 'sgd':
            self.inner_lr = self.cfg.meta.inner_lr      # if vanilla-sgd is used, inner_lr is a constant scalar
            optim_params = self.model.parameters()
        else:
            raise NotImplementedError("Unknown inner optim in cfg - use sgd or meta-sgd.")

        # outer-loop optim:
        optim = torch.optim.Adam(params=optim_params,
                                 lr=self.cfg.meta.outer_lr,
                                 weight_decay=self.cfg.meta.weight_decay)
        return optim

    def log_loss(self, currentEpoch, loss, epoch_start):
        self.writer.add_scalar('MetaLoss', loss, global_step=currentEpoch + 1)
        print("Epoch {} - MetaLoss: {:.5f} - "
              "EpochTime: {:.5f}s".format(currentEpoch + 1, loss, time.time() - epoch_start))

    ############################################################
    # Saving / Loading Routines
    ############################################################

    def save_checkpoint(self, currentEpoch, train_loss, test_loss, name=None):
        checkpt = {'model_state_dict': self.model.state_dict(),
                   'optim_state_dict': self.meta_optim.state_dict(),
                   'epoch': currentEpoch,
                   'trainLoss': train_loss,
                   'testLoss': test_loss,
                   'param_lr': self.param_lr}  # if meta-sgd is not used, this will be an empty dict
        filename = 'ep{}.pt'.format(currentEpoch) if name is None else name
        torch.save(checkpt, filename)

    def load_inneroptim(self, checkpt):
        if self.cfg.meta.inner_optim == 'meta_sgd':
            self.param_lr = checkpt['param_lr']
        print("Loaded {} weights from {}".format(self.cfg.meta.inner_optim, self.checkpt_path))

    def load_weights(self, checkpt):
        self.model.load_state_dict(checkpt['model_state_dict'])
        print("Loaded model weights from {}".format(self.checkpt_path))

    def load_checkpt(self, checkpt_path):
        checkpt = torch.load(checkpt_path)
        self.load_weights(checkpt)
        self.load_inneroptim(checkpt)

    ############################################################
    # Application Routines
    ############################################################

    def update_parameters(self, loss, used_params, second_order_bw=True):
        # calculate the gradient dLoss/dused_params and apply sgd or meta-sgd

        zero_gradients(used_params.values())
        grads = torch.autograd.grad(loss, used_params.values(), create_graph=second_order_bw, retain_graph=True)

        # when using meta-sgd, update each param with its own (learnable) learning rate
        for (k, v), grad in zip(used_params.items(), grads):
            lr = self.param_lr[k] if self.cfg.meta.inner_optim == 'meta_sgd' else self.inner_lr
            used_params[k] = v - lr * grad

        # used_params are now updated
        return used_params

    def execute_fw_pass(self, model, params, model_input, gt, return_output=False):
        # execute a forward pass.
        # params is a weight dict we currently optimize and which will be used instead of the model's own weights.
        # if this is not given, or None, the model will use its internal weights to perform the fw pass.

        model_pred = model(model_input, params)
        if return_output:
            return self.loss_fn(model_input, model_pred, gt), model_pred
        else:
            return self.loss_fn(model_input, model_pred, gt)

    def execute_inner_loop(self, task, model=None, mode='train'):
        # execute the Metappearance innerloop: perform k steps of gradient descent with the initial model weights

        if model is None:
            # if a specific model is provided, use this for the inner loop, else, use own model
            model = self.model

        # intial model weights
        adapted_params = {k: v for k, v in model.named_parameters()}

        # for fomaml or reptile, no 2nd order data should be backpropagated. also, during eval/test we want simple SGD.
        allow_second_order = True if (self.cfg.meta.algorithm == 'maml' and mode == 'train') else False

        if self.cfg.general.mode == 'brdf':
            task.shuffle()

        for k in range(self.cfg.meta.num_inner_steps):
            model_in, gt = task.sample()
            loss = self.execute_fw_pass(model=model,
                                        params=adapted_params,
                                        model_input=model_in, gt=gt)

            # when first iteration: link loss-calculation to model parameters, necessary for autograd
            p = {k: v for k, v in model.named_parameters()} if k == 0 else adapted_params

            # update parameters with a gradient descent step
            adapted_params = self.update_parameters(loss=loss,
                                                    used_params=p,
                                                    second_order_bw=allow_second_order)

        # with final params, after all gradient updates:
        model_in, gt = task.sample(mode='test')
        final_loss = self.execute_fw_pass(model=model,
                                          params=adapted_params,
                                          model_input=model_in, gt=gt)

        return final_loss, adapted_params

    def execute_outer_loop(self):
        # main training loop: for n epochs, sample a task, perform the innerloop with it, calc. meta-gradients,
        # and update model init and param_lr.

        for epoch in range(self.cfg.train.epochs):
            epoch_starttime = time.time()
            self.meta_optim.zero_grad()

            # inner loop batch
            meta_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for b in range(self.cfg.meta.num_inner_tasks):
                task = self.trainDistr.sample_task()
                task_loss, _ = self.execute_inner_loop(task=task)
                meta_loss = meta_loss + task_loss

            meta_loss = meta_loss / float(self.cfg.meta.num_inner_tasks)

            meta_loss.backward()
            self.meta_optim.step()

            # logging:
            if (epoch + 1) % self.cfg.train.logLoss_every_n == 0:
                self.log_loss(currentEpoch=epoch, loss=meta_loss.item(), epoch_start=epoch_starttime)

            # evaluate model, save weights if necessary
            if (epoch + 1) % self.cfg.train.eval_every_n == 0:

                with torch.no_grad():
                    train_tasks = [self.trainDistr[k] for k in range(len(self.trainDistr))]
                    train_loss, _ = self.evaluate_model(tasks=train_tasks, epoch=epoch + 1,
                                                        full_final=self.cfg.train.full_final, save_converged=False)

                    test_tasks = [self.testDistr[k] for k in range(len(self.testDistr))]
                    test_loss, _ = self.evaluate_model(tasks=test_tasks, epoch=epoch + 1,
                                                       full_final=self.cfg.train.full_final, save_converged=False)

                    print("Evaluated Model after {} Epochs - "
                          "Avg. Trainloss {:.5f}, Avg. Testloss {:.5f}".format(epoch + 1, train_loss, test_loss))

                    self.writer.add_scalar('Loss/train', train_loss, epoch + 1)
                    self.writer.add_scalar('Loss/test', test_loss, epoch + 1)

                if test_loss < self.bestLoss:
                    self.bestLoss = test_loss
                    self.save_checkpoint(currentEpoch=epoch + 1,
                                         train_loss=train_loss,
                                         test_loss=test_loss,
                                         name='best.pt')

        print("Done training.")

    def evaluate_model(self, tasks, epoch, full_final, save_converged=False, save_output=False):
        """
        for each task in tasks: do k steps of GD on this task to get the final task-specific model weights

        args:
         full_final: calculate the loss on the full task once the innerloop is complete to avoid noisy batches.
         save_converged: for each task, save a converged model instance.
         save_output: for each task, save the converged model's output, e.g., an image of a texture.
        """

        # detach model for evaluation, to keep weights unchanged
        model_clone = copy.deepcopy(self.model)

        results = {}
        for idx, task in enumerate(tasks):

            # detach again, per-task, to keep detached model unchanged
            model = copy.deepcopy(model_clone)

            # evaluate is called within torch.no_grad() for higher efficiency, but innerloop needs grad
            with torch.enable_grad():
                loss, final_params = self.execute_inner_loop(task,
                                                             model=model,
                                                             mode='test')

            if full_final:
                # get final loss: go through all tasks samples with the final model --> less noisy
                assert self.cfg.general.mode == 'brdf', 'FullFinal only makes sense for sample-based tasks (brdf).'
                final_loss = self.process_final_task(task=task, model=model, params=final_params)
            else:
                model_in, gt = task.sample()
                final_loss = self.execute_fw_pass(model, final_params, model_in, gt)

            # get final output with the updated parameters
            model_in, gt = task.sample()
            loss, final_output = self.execute_fw_pass(model, final_params, model_in, gt, return_output=True)

            results[task.id] = final_loss.detach().cpu().item()
            results[task.id + '_output'] = final_output.detach().cpu().numpy()

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(final_output.detach().cpu().squeeze().permute(1, 2, 0))
            ax[0].set_title('Final')
            ax[1].imshow(gt.detach().squeeze().cpu().permute(1, 2, 0))
            ax[1].set_title('GT')
            [a.axis('off') for a in ax]
            plt.title('Task {} - {}'.format(task.id, final_loss.item()))
            plt.show()

            if save_converged is True:
                # make one folder per task:
                path_for_task = os.path.join(self.converged_savedir, task.id)
                if not os.path.exists(path_for_task): os.mkdir(path_for_task)
                self.save_model(filepath=os.path.join(path_for_task, 'model.pt'), params=final_params, epoch=epoch)
                print("Saved converged models to {}".format(path_for_task))

            if save_output is True:
                assert self.cfg.general.mode != 'brdf', "BRDF Output should be saved as network per task. " \
                                                      "Set save_converged instead of save_output."
                torch.save(results, os.path.join(self.output_savedir, 'output_ep{}.pt'.format(epoch)))
                torchvision.io.write_png((final_output.clamp(0.0, 1.0) * 255.0).byte().squeeze().cpu(),
                                          os.path.join(self.output_savedir, task.id+'.png'))
                print("Saved output to {}".format(self.output_savedir))

        assert len([results[x] for x in results.keys() if not 'output' in x]) == len(tasks)
        avg_loss = sum([results[x] for x in results.keys() if not 'output' in x]) / len(tasks)
        return avg_loss, results

    def process_final_task(self, task, model, params):
        # for sampling-based tasks, e.g., brdf: process the full task instead of just #batchsize samples.
        losses = torch.tensor(0.0, device=self.device)
        num_batches = int(task.test_samples.shape[0] / self.cfg.train.batchsize)
        for j in range(num_batches):
            model_in, gt = task.get_testbatch(j * self.cfg.train.batchsize)
            loss = self.execute_fw_pass(model=model,
                                        params=params,
                                        model_input=model_in, gt=gt)
            losses += loss
        return losses / num_batches

    def save_model(self, filepath, params, epoch):
        if self.cfg.general.mode == 'brdf':
            # special case, bc mitsuba expects a certain format for rendering
            self.model.save_to_npy(epoch=epoch, savepath=filepath, params=params)
        else:
            checkpt = {'model_state_dict': params,
                       'param_lr': self.param_lr}
            torch.save(checkpt, filepath)
