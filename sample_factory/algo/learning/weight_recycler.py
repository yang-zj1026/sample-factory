import copy
import functools
import torch
import torch.nn as nn


def reset_momentum(momentum, mask):
    new_momentum = momentum if mask is None else momentum * (1.0 - mask)
    return new_momentum


def weight_reinit_zero(param, mask):
    if mask is None:
        return param
    else:
        new_param = torch.zeros_like(param)
        param = torch.where(mask == 1, new_param, param)
        return param


def weight_reinit_random(param, mask, weights_type='incoming', weight_scaling=False, scale=1.0):
    """Randomly reinit recycled weights and may scale its norm.

    If scaling applied, the norm of recycled weights equals
    the average norm of non-recycled weights per neuron multiplied by a scalar.

    Args:
        param: current param
        mask: incoming/outgoing mask for recycled weights
        weight_scaling: if true, scale recycled weights with the norm of non-recycled
        scale: scale to multiply the new weights norm.
        weights_type: incoming or outgoing weights

    Returns:
        params: new params after weight recycle.
    """

    new_param = nn.init.xavier_uniform_(torch.empty_like(param)).to(param.device)

    if weight_scaling:
        axes = list(range(param.ndim))
        if weights_type == 'outgoing':
            del axes[-2]
        else:
            del axes[-1]

        neuron_mask = torch.mean(mask, dim=axes)

        non_dead_count = neuron_mask.shape[0] - torch.count_nonzero(neuron_mask)
        norm_per_neuron = _get_norm_per_neuron(param, axes)
        non_recycled_norm = torch.sum(norm_per_neuron * (1 - neuron_mask)) / non_dead_count
        non_recycled_norm = non_recycled_norm * scale

        normalized_new_param = _weight_normalization_per_neuron_norm(new_param, axes)
        new_param = normalized_new_param * non_recycled_norm

    param = torch.where(mask == 1, new_param, param)
    return param


def _weight_normalization_per_neuron_norm(param, axes):
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    norm_per_neuron = norm_per_neuron.unsqueeze(dim=axes)
    normalized_param = param / norm_per_neuron
    return normalized_param


def _get_norm_per_neuron(param, axes):
    return torch.sqrt(torch.sum(torch.pow(param, 2), dim=axes))


class BaseRecycler:
    """
    Base class for weight update methods.

    Attributes:
        all_layers_names: list of layer names in a model.
        recycle_type: neuron, layer based.
        dead_neurons_threshold: below this threshold a neuron is considered dead.
        reset_layers: list of layer names to be recycled.
        reset_start_layer_idx: index of the layer from which we start recycling.
        reset_period: int represents the period of weight update.
        reset_start_step: start recycle from start step
        reset_end_step:  end recycle from end step
        logging_period:  the period of statistics logging e.g., dead neurons.
        prev_neuron_score: score at last reset step or log step in case of no reset.
        sub_mean_score: if True the average activation will be subtracted
                        for each neuron when we calculate the score.
    """

    def __init__(self, all_layers_names, dead_neurons_threshold=0.0, reset_start_layer_idx=0, reset_period=200_000,
                 reset_start_step=0, reset_end_step=100_000_000, logging_period=20_000, sub_mean_score=False):
        self.all_layers_names = all_layers_names
        self.dead_neurons_threshold = dead_neurons_threshold
        self.reset_layers = all_layers_names[reset_start_layer_idx:]
        self.reset_period = reset_period
        self.reset_start_step = reset_start_step
        self.reset_end_step = reset_end_step
        self.logging_period = logging_period
        self.prev_neuron_score = None
        self.sub_mean_score = sub_mean_score

        self._last_update_step = None

    def update_reset_layers(self, reset_start_layer_idx):
        self.reset_layers = self.all_layers_names[reset_start_layer_idx:]

    def is_update_iter(self, step):
        return step > 0 and (step % self.reset_period == 0)

    def update_weights(self, intermediates, params, opt_state):
        raise NotImplementedError

    def maybe_update_weights(self, update_step, intermediates, params, opt_state):
        self._last_update_step = update_step
        if self.is_reset(update_step):
            new_params, new_opt_state = self.update_weights(intermediates, params, opt_state)
        else:
            new_params, new_opt_state = params, opt_state

        return new_params, new_opt_state

    def is_reset(self, update_step):
        del update_step
        return False

    def is_intermediated_required(self, update_step):
        return self.is_logging_step(update_step)

    def is_logging_step(self, step):
        return step % self.logging_period == 0

    def maybe_log_deadneurons(self, update_step, intermediates):
        is_logging = self.is_logging_step(update_step)
        if is_logging:
            return self.log_dead_neurons_count(intermediates)
        else:
            return None

    def intersected_dead_neurons_with_last_reset(self, intermediates, update_step):
        if self.is_logging_step(update_step):
            log_dict = self.log_intersected_dead_neurons(intermediates)
            return log_dict
        else:
            return None

    def log_intersected_dead_neurons(self, intermediates):
        """
        Track intersected dead neurons with last logging/reset step.

        Args:
          intermediates: current intermediates

        Returns:
          log_dict: dict contains the percentage of intersection
        """

        neuron_score_dict = {key: self.estimate_neuron_score(intermediates[key]) for key in self.reset_layers}

        if self.prev_neuron_score is None:
            self.prev_neuron_score = neuron_score_dict
            log_dict = None
        else:
            log_dict = {}
            for prev_k_score, current_k_score in zip(self.prev_neuron_score.items(), neuron_score_dict.items()):
                prev_k, prev_score = prev_k_score
                current_k, score = current_k_score
                prev_mask = prev_score <= self.dead_neurons_threshold
                intersected_mask = prev_mask & (score <= self.dead_neurons_threshold)
                prev_dead_count = torch.count_nonzero(prev_mask)
                intersected_count = torch.count_nonzero(intersected_mask)

                percent = (float(intersected_count) / prev_dead_count) if prev_dead_count else 0.0
                log_dict[f'dead_intersected_percent/{current_k}'] = float(percent) * 100.

                nondead_mask = score > self.dead_neurons_threshold

                log_dict[f'mean_score_recycled/{current_k}'] = float(torch.mean(score[prev_mask]))
                log_dict[f'mean_score_nondead/{current_k}'] = float(torch.mean(score[nondead_mask]))

            self.prev_neuron_score = neuron_score_dict

        return log_dict

    def log_dead_neurons_count(self, intermediates):
        """
        log dead neurons in each layer.

        For conv layer we also log dead elements in the spatial dimension.

        Args:
          intermediates: intermidate activation in each layer.

        Returns:
          log_dict_elements_per_neuron
          log_dict_neurons
        """

        def log_dict(score_dict):
            total_neurons, total_deadneurons = 0., 0.

            log_dict = {}
            for k, m in score_dict.items():
                layer_size = float(torch.numel(m))
                deadneurons_count = torch.count_nonzero(m <= self.dead_neurons_threshold).item()
                total_neurons += layer_size
                total_deadneurons += deadneurons_count

            log_dict[f'Dormant/total'] = total_neurons
            log_dict[f'Dormant/deadcount'] = float(total_deadneurons)
            log_dict[f'Dormant/dormant_percentage'] = (float(total_deadneurons) / total_neurons)
            return log_dict

        neuron_score = {k: self.estimate_neuron_score(intermediates[k]) for k in self.reset_layers}
        log_dict_neurons = log_dict(neuron_score)

        return log_dict_neurons

    def estimate_neuron_score(self, activation):
        """
        Calculates neuron score based on absolute value of activation.

        The score of feature map is the normalized average score over
        the spatial dimension.

        Args:
          activation: intermediate activation of each layer
          is_cbp: if true, subtracts the mean and skips normalization.

        Returns:
          element_score: score of each element in feature map in the spatial dim.
          neuron_score: score of feature map
        """

        reduce_axes = list(range(activation.ndim - 1))
        if self.sub_mean_score:
            activation = activation - torch.mean(activation, dim=reduce_axes)

        score = torch.mean(torch.abs(activation), dim=reduce_axes)
        # Normalize so that all scores sum to one.
        score /= torch.sum(score) + 1e-9

        return score


class NeuronRecycler(BaseRecycler):
    """
    Recycle the weights connected to dead neurons.

    In convolutional neural networks, we consider a feature map as neuron.

    Attributes:
        next_layers: dict key a current layer name, value next layer name.
        init_method_outgoing: method to init outgoing weights (random, zero).
        weight_scaling: if true, scale reinit weights.
        incoming_scale: scalar for incoming weights.
        outgoing_scale: scalar for outgoing weights.
    """

    def __init__(self, all_layers_names, reset_layer_names, reset_layer_idx, next_layers, init_method_outgoing='zero',
                 weight_scaling=False, incoming_scale=1.0, outgoing_scale=1.0,
                 reset_period=200_000):
        super(NeuronRecycler, self).__init__(all_layers_names, reset_period=reset_period)
        self.init_method_outgoing = init_method_outgoing
        self.weight_scaling = weight_scaling
        self.incoming_scale = incoming_scale
        self.outgoing_scale = outgoing_scale
        # prepare a dict that has pointer to next layer give a layer name
        # this is needed because neuron recycle reinitalizes both sides
        # (incoming and outgoing weights) of a neuron and needs a point to the
        # outgoing weights.
        self.next_layers = next_layers
        self.next_layers_keys = list(next_layers.keys())

        # recycle the neurons in the given layer.
        self.reset_layers = reset_layer_names
        self.reset_layers_idx = reset_layer_idx

    def intersected_dead_neurons_with_last_reset(self, intermediates,
                                                 update_step):
        if self.is_reset(update_step):
            log_dict = self.log_intersected_dead_neurons(intermediates)
            return log_dict
        else:
            return None

    def is_reset(self, update_step):
        within_reset_interval = (self.reset_start_step <= update_step < self.reset_end_step)
        return self.is_update_iter(update_step) and within_reset_interval

    def is_intermediated_required(self, update_step):
        is_logging = self.is_logging_step(update_step)
        is_update_iter = self.is_update_iter(update_step)
        return is_logging or is_update_iter

    def update_weights(self, intermediates, params, opt_state):
        new_param, opt_state = self.recycle_dead_neurons(intermediates, params, opt_state)
        return new_param, opt_state

    def recycle_dead_neurons(self, intermedieates, params_dict, opt_state):
        """Recycle dead neurons by reinitializing incoming and outgoing connections.

        Incoming connections are randomly initialized and outgoing connections
        are zero initialized.
        A featuremap is considered dead when its score is below or equal
        dead neuron threshold.
        Args:
          intermedieates: pytree contains the activations over a batch.
          params: current weights of the model.
          key: used to generate random keys.
          opt_state: state of optimizer.

        Returns:
          new model params after recycling dead neurons.
          opt_state: new state for the optimizer

        Raises: raise error if init_method_outgoing is not one of the following
        (random, zero).
        """
        activations_score_dict = {k: intermedieates[k] for k in self.reset_layers}

        # create incoming and outgoing masks and reset bias of dead neurons.
        (
            incoming_mask_dict,
            outgoing_mask_dict,
            params_dict,
        ) = self.create_masks(params_dict, activations_score_dict)

        param_state = opt_state['state']

        # reset incoming weights, bias and momentum of optimizer
        for reset_layer in self.reset_layers:
            param_key = reset_layer + '.weight'
            param = params_dict[param_key]
            incoming_mask = incoming_mask_dict[param_key]
            params_dict[param_key] = weight_reinit_random(param, incoming_mask, weights_type='incoming')

            param_idx = self.reset_layers_idx[param_key]
            param_state[param_idx]['exp_avg'] = reset_momentum(param_state[param_idx]['exp_avg'],
                                                               incoming_mask_dict[param_key])
            param_state[param_idx]['exp_avg_sq'] = reset_momentum(param_state[param_idx]['exp_avg_sq'],
                                                                  incoming_mask_dict[param_key])

            bias_key = reset_layer + '.bias'
            bias = params_dict[bias_key]
            incoming_mask = incoming_mask_dict[bias_key]
            params_dict[bias_key] = weight_reinit_zero(bias, incoming_mask)

            bias_idx = self.reset_layers_idx[bias_key]
            param_state[bias_idx]['exp_avg'] = reset_momentum(param_state[bias_idx]['exp_avg'],
                                                              incoming_mask_dict[bias_key])
            param_state[bias_idx]['exp_avg_sq'] = reset_momentum(param_state[bias_idx]['exp_avg_sq'],
                                                                 incoming_mask_dict[bias_key])

        # reset outgoing weights and momentum of optimizer
        for reset_layer in self.reset_layers:
            if reset_layer in self.next_layers_keys:
                next_layer = self.next_layers[reset_layer]
                next_param_key = next_layer + '.weight'
                next_param = params_dict[next_param_key]
                outgoing_mask = outgoing_mask_dict[next_param_key]

                if self.init_method_outgoing == 'random':
                    params_dict[next_layer] = weight_reinit_random(next_param, outgoing_mask, weights_type='outgoing')
                elif self.init_method_outgoing == 'zero':
                    params_dict[next_layer] = weight_reinit_zero(next_param, outgoing_mask)
                else:
                    raise ValueError(f'Invalid init method: {self.init_method_outgoing}')

                next_param_idx = self.reset_layers_idx[next_param_key]
                param_state[next_param_idx]['exp_avg'] = reset_momentum(param_state[next_param_idx]['exp_avg'],
                                                                        outgoing_mask_dict[next_param_key])
                param_state[next_param_idx]['exp_avg_sq'] = reset_momentum(param_state[next_param_idx]['exp_avg_sq'],
                                                                           outgoing_mask_dict[next_param_key])

        opt_state['state'] = param_state

        return params_dict, opt_state

    def _score2mask(self, activation):
        score = self.estimate_neuron_score(activation)
        mask = score <= self.dead_neurons_threshold
        return mask.float()

    def create_masks(self, param_dict, activations_dict):
        incoming_mask_dict = {
            k: torch.zeros_like(p) if p.dim() != 1 else None
            for k, p in param_dict.items()
        }
        outgoing_mask_dict = {
            k: torch.zeros_like(p) if p.dim() != 1 else None
            for k, p in param_dict.items()
        }

        # Prepare mask of incoming and outgoing recycled connections
        for reset_layer in self.reset_layers:
            param_key = reset_layer + '.weight'
            param = param_dict[param_key]
            activation = activations_dict[reset_layer]
            neuron_mask = self._score2mask(activation)

            # Create weight mask
            next_param = None
            next_param_key = None
            if reset_layer in self.next_layers_keys:
                next_param_key = self.next_layers[reset_layer] + '.weight'
                next_param = param_dict[next_param_key]
            incoming_mask, outgoing_mask = self.create_mask_helper(neuron_mask, param, next_param)
            incoming_mask_dict[param_key] = incoming_mask
            if next_param_key:
                outgoing_mask_dict[next_param_key] = outgoing_mask

            # Create bias mask
            bias_key = reset_layer + '.bias'
            incoming_mask_dict[bias_key] = neuron_mask

        return (
            incoming_mask_dict,
            outgoing_mask_dict,
            param_dict
        )

    def mask_creator(self, expansion_axis, param, neuron_mask):
        """Create a mask of weight matrix given 1D vector of neurons mask.

        Args:
            expansion_axis: List containing 1 axis. The dimension to expand the mask
                for dense layers (weight shape 2D).
            param: Weight.
            neuron_mask: 1D mask that represents dead neurons (features).

        Returns:
            mask: Mask of weight.
        """

        if expansion_axis == 0:
            # Incoming weights
            mask = neuron_mask.unsqueeze(1)
            mask = mask.expand(-1, param.shape[1])
        else:
            # Outgoing weights
            mask = neuron_mask.unsqueeze(0)
            mask = mask.expand(param.shape[0], -1)

        return mask

    def create_mask_helper(self, neuron_mask, current_param, next_param):
        """Generate incoming and outgoing weight mask given dead neurons mask.

        Args:
            neuron_mask: Mask of size equal to the width of a layer.
            current_param: Incoming weights of a layer.
            next_param: Outgoing weights of a layer.

        Returns:
            incoming_mask
            outgoing_mask
        """

        incoming_mask = self.mask_creator(0, current_param, neuron_mask)
        outgoing_mask = None
        if next_param is not None:
            outgoing_mask = self.mask_creator(1, next_param, neuron_mask)
        return incoming_mask, outgoing_mask
