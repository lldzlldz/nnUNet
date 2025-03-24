import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class nnUNetPlannerAvgSpacingPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlannerAvgSpacingPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        # self.UNet_class = PlainConvUNet
        # the following two numbers are really arbitrary and were set to reproduce default nnU-Net's configurations as
        # much as possible
        # self.plans_identifier = 'nnUNetPlans_avg_spacing'


    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name
    
    def determine_fullres_target_spacing(self) -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        if self.overwrite_target_spacing is not None:
            return np.array(self.overwrite_target_spacing)

        spacings = np.vstack(self.dataset_fingerprint['spacings'])
        sizes = self.dataset_fingerprint['shapes_after_crop']

        # target = np.percentile(spacings, 50, 0)

        # # todo sizes_after_resampling = [compute_new_shape(j, i, target) for i, j in zip(spacings, sizes)]

        # target_size = np.percentile(np.vstack(sizes), 50, 0)
        # # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # # the following properties:
        # # - one axis which much lower resolution than the others
        # # - the lowres axis has much less voxels than the others
        # # - (the size in mm of the lowres axis is also reduced)
        # worst_spacing_axis = np.argmax(target)
        # other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        # other_spacings = [target[i] for i in other_axes]
        # other_sizes = [target_size[i] for i in other_axes]

        # has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        # has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        # if has_aniso_spacing and has_aniso_voxels:
        #     spacings_of_that_axis = spacings[:, worst_spacing_axis]
        #     target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        #     # don't let the spacing of that axis get higher than the other axes
        #     if target_spacing_of_that_axis < max(other_spacings):
        #         target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        #     target[worst_spacing_axis] = target_spacing_of_that_axis

        average = np.mean(spacings)
        average_array = np.array([average, average, average])
        
        

        return average_array
    



    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name
