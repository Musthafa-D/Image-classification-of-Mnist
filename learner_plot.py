import torch
import ccbdl
import os
from plots import *
from ccbdl.learning.gan import BaseDifussionLearning
from ccbdl.utils import DEVICE
import sys
import time


class Learner(BaseDifussionLearning):
    def __init__(self,
                 result_folder,
                 model,
                 train_data,
                 test_data,
                 val_data,
                 config,
                 network_config,
                 logging):
        """
        init function of the learner class.
        
        Args:
            model : The network that you use.
                --> Example CNN, FNN, etc.
            
            train_data, test_data val_data: Respective 
            train, test data and val_data.
                --> Example like Cifar10's train
                    and test data.

        Returns
            None.
        """
        super(Learner, self).__init__(train_data, test_data,
                                      val_data, result_folder, config, logging=logging)
        self.device = DEVICE
        print(self.device)
        
        self.model = model
        
        self.figure_storage.dpi=600
        
        self.learner_config = config
        self.network_config = network_config

        self.result_folder = result_folder
        
        #self.plotter.register_default_plot(Softmax_plot_update(self, "seperate"))
        #self.plotter.register_default_plot(Softmax_plot_update(self, "combined"))
        #self.plotter.register_default_plot(Attribution_plots_update(self, "seperate")) 
        #self.plotter.register_default_plot(Attribution_plots_update(self, "combined")) 
        #if self.network_config["final_layer"] == 'nlrl':
        	#self.plotter.register_default_plot(Hist_plot_update(self))
        self.plotter.register_default_plot(Tsne_plot_update(self))

        self.parameter_storage.store(self)
        self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters: ")
        self.parameter_storage.write_tab(self.model.count_learnable_parameters(), 
                                         "number of learnable parameters: ")
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')
        
        # for name, param in model.named_parameters():
        #     print(name)
        
        # Replace DataStorage store method with store_new for calculating correct a_train_Acc and a_train_loss
        self.data_storage.store = self.store_new

    def _train_epoch(self, train=True):
        if self.logging:
            self.logger.info("started epoch %i." % self.epoch)

        self.model.train()

        for i, data in enumerate(self.train_data):            
            inputs, labels = data
            inputs, labels = inputs.to(
                self.device), labels.to(self.device).long()

            if train:
                self.batch += 1  
                
                self.data_storage.dump_store("train_inputs", inputs)
                self.data_storage.dump_store("train_actual_label", labels)

    def _test_epoch(self):        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                images, labels = data
                images, labels = images.to(
                    self.device), labels.to(self.device).long()
                
                self.data_storage.dump_store("test_inputs", images)
                self.data_storage.dump_store("test_actual_label", labels)

    def _validate_epoch(self):
        pass

    def noising_images(self):
        pass
    
    def noise_prediction(self):
        pass


    def _update_best(self):
        pass

    def evaluate(self):
        pass

    def _hook_every_epoch(self):
        print(f"Epoch: {self.epoch}")
        self.data_storage.dump_store("epochs_gen", self.epoch)
            
    def _save(self):
        pass
    
    def _load_initial(self):
        if self.network_config["final_layer"] == "nlrl":
            checkpoint_path = os.path.join("Networks_plot", "net_initial.pt")
        else:
            checkpoint_path = os.path.join("Networks_plot", "net_initial_linear.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
        except PermissionError as e:
            raise PermissionError(f"Permission denied: {checkpoint_path}") from e
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
    
    def _load_best(self):
        if self.network_config["final_layer"] == "nlrl":
            checkpoint_path = os.path.join("Networks_plot", "net_best.pt")
        else:
            checkpoint_path = os.path.join("Networks_plot", "net_best_linear.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
        except PermissionError as e:
            raise PermissionError(f"Permission denied: {checkpoint_path}") from e
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
    
    def store_new(self, vals, force=False):
        """
        New store method to replace the default store method for DataStorage.

        Parameters
        ----------
        vals : list of values
            List of values to be stored in the internal 'stored_values'-dictionary.\n
            Order has to be the same as given during initialization. Best used with \n
            int, float or torch.Tensor.
        force : int
            If given an integer it appends the values with the given batch number.

        Returns
        -------
        None.

        """
        data_storage = self.data_storage  # Reference to data_storage
        # save time when first storing
        if data_storage.batch == 0:
            data_storage.dump_values["TimeStart"] = time.time()
        if data_storage.batch % data_storage.step == 0 or force > 0:
            if len(data_storage.stored_values["Time"]) == 0:
                data_storage.stored_values["Time"] = [
                    (time.time() - data_storage.dump_values["TimeStart"]) / 60]
            else:
                data_storage.stored_values["Time"].append(
                    (time.time() - data_storage.dump_values["TimeStart"]) / 60.0)
            for col in range(1, data_storage.columns):
                name = data_storage.names[col]
                if name == "a_train_loss":
                    if len(data_storage.stored_values["train_loss"]) < data_storage.average_window:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_loss"]))
                    else:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_loss"][-data_storage.average_window:]))
                    data_storage.stored_values[name].append(avg)
                elif name == "a_train_acc":
                    if len(data_storage.stored_values["train_acc"]) < data_storage.average_window:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_acc"]))
                    else:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_acc"][-data_storage.average_window:]))
                    data_storage.stored_values[name].append(avg)
                else:
                    if type(vals[col - 1]) == torch.Tensor:
                        data_storage.stored_values[name].append(
                            vals[col - 1].cpu().detach().item())
                    else:
                        data_storage.stored_values[name].append(vals[col - 1])
    
            if data_storage.batch == 0:
                data_storage._get_head()
                data_storage._display()
                print("")
            else:
                if data_storage.batch % data_storage.show == 0 or force > 0:
                    data_storage._display()
                if data_storage.batch % data_storage.line == 0:
                    print("")
                if data_storage.batch % data_storage.header == 0:
                    data_storage._get_head()
        data_storage.batch += 1
