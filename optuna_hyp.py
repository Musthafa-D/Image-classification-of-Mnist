from learner import Learner
from networks import CNN
from ccbdl.utils import DEVICE
from ccbdl.parameter_optimizer.optuna_base import BaseOptunaParamOptimizer
from datetime import datetime, timedelta
from metrics import Metrics
from data_loader import prepare_data
import optuna
import ccbdl
import os
import matplotlib.pyplot as plt
import torch
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


class Optuna(BaseOptunaParamOptimizer):
    def __init__(self,
                 study_config: dict,
                 optimize_config: dict,
                 network_config: dict,
                 data_config: dict,
                 learner_config: dict,
                 config: dict,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        """
        init function of the study.
        
        Args:
            study_config, optimize_config, network_config,
            data_config, learner_config, and config: dict
                --> respective values from the respective
                    headings from the config file.
                --> argument, config holds represents the entire
                    config file.
            
            study_path, config_path : str
                --> respective paths of study and config.
            
            comment : str
                --> adding comments for the study.
                --> default is "".
            
            debug, logging : bool
               --> sets the debug and logging of the study
                   if True.
               --> default is False.

        Returns
            None.
        """

        if "sampler" in study_config.keys():
            if hasattr(optuna.samplers, study_config["sampler"]["name"]):
                sampler = getattr(
                    optuna.samplers, study_config["sampler"]["name"])()
        else:
            sampler = optuna.samplers.TPESampler()

        if "pruner" in study_config.keys():
            if hasattr(optuna.pruners, study_config["pruner"]["name"]):
                pruner = getattr(
                    optuna.pruners, study_config["pruner"]["name"])()
        else:
            pruner = None
        
        super().__init__(study_config["direction"], study_config["study_name"], study_path,
                         study_config["number_of_trials"], data_config["task"], comment, 
                         study_config["optimization_target"],
                         sampler, pruner, config_path, debug, logging)
        self.optimize_config = optimize_config
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.study_config = study_config
        self.result_folder = study_path
        self.config = config

        optuna.logging.disable_default_handler()

        self.create_study()
        self.study_name = study_config["study_name"]
        self.durations = []
        
        self.learnable_parameters_list = []
        self.optimization_target_values = []
        self.lr_list = []
        self.overall_duration_metrics_list = []

    def _objective(self, trial):
        """
        objective function of the study.
        
        Args:
            trial: (optuna.trial.Trial)
                --> current optimization trial of
                    the study.

        Returns
            the best value of the optimization target.
                --> eg: like maximum test accuracy.
        """
        start_time = datetime.now()

        if self.logging:
            self.logger.info("start trial %i" % trial.number)

        print("\n\n******* Trial " + str(trial.number) +
              " has started" + "*******\n")

        trial_folder = f'trial_{trial.number}'
        trial_path = os.path.join(self.result_folder, trial_folder)
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)

        # suggest parameters
        suggested = self._suggest_parameters(self.optimize_config, trial)
        self.learner_config["learning_rate_exp"] = suggested["learning_rate_exp"]
        self.learner_config["weight_decay_rate"] = suggested["weight_decay_rate"]
        self.network_config["num_blocks"] = suggested["num_blocks"]

        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)

        if self.learner_config["cnn_model"] == 'grayscale':
            # Classifier
            network = CNN(1,"Classifier", **self.network_config).to(DEVICE)
        elif self.learner_config["cnn_model"] == 'rgb':
            # Classifier
            network = CNN(3,"Classifier_rgb", **self.network_config).to(DEVICE)
        else:
            raise ValueError("Invalid values, it's either grayscale or rgb")

        self.learner = Learner(model=network,
                               train_data=train_data,
                               test_data=test_data,
                               val_data=val_data,
                               config=self.learner_config,
                               network_config=self.network_config,
                               result_folder=trial_path,
                               logging=True)
        
        self.learner.parameter_storage.write("Current config:-")
        self.learner.parameter_storage.store(self.config)

        self.learner.fit(test_epoch_step=self.learner_config["testevery"])

        self.learner.parameter_storage.write(
            f"Start Time of training and evaluation of the dataset in this Trial {trial.number}: {start_time.strftime('%H:%M:%S')}")

        self.learner.parameter_storage.store(
            suggested, header="suggested_parameters")

        self.learner.parameter_storage.write("\n")

        if self.logging:
            self.logger.info("finished trial")
        print(f"\n\n******* Trial {trial.number} is completed*******")

        end_time = datetime.now()
        self.learner.parameter_storage.write(
            f"End Time of training and evaluation of the dataset in this Trial {trial.number}: {end_time.strftime('%H:%M:%S')}\n")

        self.duration_trial = end_time - start_time
        self.durations.append(self.duration_trial)
        self.learner.parameter_storage.write(
            f"Duration of training and evaluation of the dataset in this Trial Trial {trial.number}: {str(self.duration_trial)[:-7]}\n")
        
        # Storing the number of learnable parameters for this trial
        learnable_params = self.learner.model.count_learnable_parameters()
        self.learnable_parameters_list.append(learnable_params)
        
        # Storing the learning rate for this trial
        lr_params = self.learner.learning_rate
        self.lr_list.append(lr_params)
        
        # Storing the optimization target value for this trial
        optim_target_val = self.learner.best_values[self.optimization_target]
        self.optimization_target_values.append(optim_target_val)

        return self.learner.best_values[self.optimization_target]

    def start_study(self):
        """
        start_study function of the study
            --> starts the study with the objective and number 
                of trials
        Returns
            None.
        """
        self.study.optimize(self._objective, n_trials=self.number_of_trials,)

    def eval_study(self):
        """
        eval_study function of the study
            --> evaluates the study and provides final results
                such as optuna plots, values of the best study,
                etc.
        Returns
            None.
        """
        if self.logging:
            self.logger.info("evaluating study")
        start_time = datetime.now()
        
        parameter_storage = ccbdl.storages.storages.ParameterStorage(
            self.result_folder, file_name="study_info.txt")

        parameter_storage.write("******* Summary " +
                                "of " + self.study_name + " *******")
        pruned_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
          
        if self.logging:
            self.logger.info("creating optuna plots")

        sub_folder = os.path.join(self.result_folder, 'study_plots', 'optuna_plots')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        self.figure_storage = ccbdl.storages.storages.FigureStorage(
            sub_folder, types=("png", "pdf"))

        figures_list = []
        figures_names = []
    
        fig = optuna.visualization.plot_optimization_history(self.study)
        figures_list.append(fig)
        figures_names.append("optimization_history")
    
        fig = optuna.visualization.plot_parallel_coordinate(
            self.study, 
            params=["learning_rate_exp","weight_decay_rate","num_blocks"])
        figures_list.append(fig)
        figures_names.append("parallel_coordinate")
    
        fig = optuna.visualization.plot_param_importances(self.study)
        figures_list.append(fig)
        figures_names.append("param_importances")
    
        fig = optuna.visualization.plot_slice(
            self.study,
            params=["learning_rate_exp","weight_decay_rate","num_blocks"])
        figures_list.append(fig)
        figures_names.append("plot_slice")
    
        # Now use store_multi to store all figures at once
        self.figure_storage.store_multi(figures_list, figures_names)
        
        if self.logging:
            self.logger.info("creating accuracy vs parameters and learning plot")
            
        param_folder = os.path.join(self.result_folder, 'study_plots', 'accuracy_plots')
        if not os.path.exists(param_folder):
            os.makedirs(param_folder)
        self.fig_storage = ccbdl.storages.storages.FigureStorage(
            param_folder, types=("png", "pdf"))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(self.learnable_parameters_list, self.optimization_target_values)
        ax.set_xlabel("$|\\theta|$", fontsize=14) # number of learnable parameters
        ax.set_ylabel("$\\mathrm{Acc}$", fontsize=14)
        ax.grid(True)
        ax.set_yticks(range(0, 101, 10))
        self.fig_storage.store(fig, "param_acc_plot") # test accuracy vs learnable Parameter
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(self.lr_list, self.optimization_target_values)
        ax.set_xlabel("$lr$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Acc}$", fontsize=14)
        ax.grid(True)
        ax.set_yticks(range(0, 101, 10))
        self.fig_storage.store(fig, "lr_acc_plot") # test accuracy vs learning rate
        
        end_time = datetime.now()
        self.overall_duration = sum(self.durations, timedelta()) + (end_time - start_time)
        
        parameter_storage.write("\nStudy statistics: ")
        parameter_storage.write(
            f"  Number of finished trials: {len(self.study.trials)}")
        parameter_storage.write(
            f"  Number of pruned trials: {len(pruned_trials)}")
        parameter_storage.write(
            f"  Number of complete trials: {len(complete_trials)}")
        parameter_storage.write(
            f"  Time of study excluding metrics calculation: {str(self.overall_duration)[:-7]}")
        parameter_storage.write(
            f"\nBest trial: Nr {self.study.best_trial.number}")
        parameter_storage.write(f"  Best Value: {self.study.best_trial.value}")

        parameter_storage.write("  Params: ")
        for key, value in self.study.best_trial.params.items():
            parameter_storage.write(f"    {key}: {value}")
        parameter_storage.write("\n")

        parameter_storage.write("Parameters and their Respective Accuracies: ")
        for i in range(len(self.lr_list)):
            parameter_storage.write(f"    Trial {i}: Learning rate: {self.lr_list[i]}, Learnable Parameters: {self.learnable_parameters_list[i]}, Test Accuracy: {self.optimization_target_values[i]}\n")
        parameter_storage.write("\n")
    
    def eval_metrics(self):
        """
        eval_metrics function of the study
            --> evaluates the metric values for test_data in all trials
                of the study and provides final results such as average 
                infidelity and sensitivityof the attributions used.
        Returns
            None.
        """
        if self.logging:
            self.logger.info("calculating metrics for all trials")
        
        start_time_metircs = datetime.now()
    
        for trial in self.study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            trial_number = trial.number
            trial_folder = f"trial_{trial_number}"
            trial_path = os.path.join(self.result_folder, trial_folder)
            model_path = os.path.join(trial_path, "net_best.pt")
            
            if self.learner_config['cnn_model'] == 'grayscale':
                model = CNN(1,"Classifier",                            
                            filter_growth_rate=self.network_config['filter_growth_rate'],
                            dropout_rate=self.network_config['dropout_rate'],                         
                            final_channel=self.network_config["final_channel"],
                            activation_function=self.network_config["activation_function"],
                            initial_out_channels=self.network_config['initial_out_channels'],
                            final_layer=self.network_config['final_layer'],
                            num_blocks=trial.params['num_blocks']).to(DEVICE)
                channel = 'gray'
            
            elif self.learner_config['cnn_model'] == 'rgb':
                model = CNN(3,"Classifier_rgb",
                            filter_growth_rate=self.network_config['filter_growth_rate'],
                            dropout_rate=self.network_config['dropout_rate'],                         
                            final_channel=self.network_config["final_channel"],
                            activation_function=self.network_config["activation_function"],
                            initial_out_channels=self.network_config['initial_out_channels'],
                            final_layer=self.network_config['final_layer'],
                            num_blocks=trial.params['num_blocks']).to(DEVICE)
                channel = 'rgb'
            
            else:
                raise ValueError("Invalid values, it's either grayscale or rgb")
            
            # Load the model
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
    
            train_data, test_data, val_data = prepare_data(self.data_config)
    
            # Pass the loaded model to the Metrics class and calculate metrics
            if trial_number == self.study.best_trial.number:
                test_metrics = Metrics(model=model, test_data=test_data, result_folder=trial_path,
                                        best_trial_check=1, channel=channel)
            else:
                test_metrics = Metrics(model=model, test_data=test_data, result_folder=trial_path,
                                        best_trial_check=0, channel=channel)
            test_metrics.calculations()
            duration_metrics_per_trial = test_metrics.total_metric_duration()
            duration_per_trial = self.durations[trial_number] + duration_metrics_per_trial

            with open(os.path.join(trial_path, "ParameterStorage.txt"), "a") as file:
                file.write(f"Duration of metrics calculation of test data in this Trial {trial.number}: {str(duration_metrics_per_trial)[:-7]}\n")
                file.write(f"Total duration of this Trial: {trial.number}: {str(duration_per_trial)[:-7]}")
            
        end_time_metrics = datetime.now()
        self.duration_metrics = end_time_metrics - start_time_metircs
        with open(os.path.join(self.result_folder, "study_info.txt"), "a") as file:
            file.write(f"Time of entire metrics calculation: {str(self.duration_metrics)[:-7]}\n")
            total_duration = self.overall_duration + self.duration_metrics
            file.write(f"Time of entire study: {str(total_duration)[:-7]}")
