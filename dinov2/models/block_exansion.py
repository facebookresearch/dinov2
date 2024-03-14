import torch.nn as nn
# from drop_path import DropPath
# from layer_scale import LayerScale
from dinov2.models.layer_scale import LayerScale
from dinov2.models.drop_path import DropPath
import torch
import copy


def expand_dinov2(backbone, block_expansion_positions, block_expansion_path_dropout: float = 0.0):
    '''
    Expand the backbone of the DINOv2 model.
    :param backbone: The backbone to expand.
    :param args: The arguments of the program.
    :return: The expanded backbone.
    '''
    position_list = block_expansion_positions
    #assert all elements in list are all int
    assert all(isinstance(x, int) for x in position_list)
    #assert all elements are in range of len(backbone.blocks)
    assert all(0 <= x < len(backbone.blocks) for x in position_list)

    position_list_after_expansion = get_expanded_block_positions(position_list)
    

    # expand the backbone
    for position in position_list_after_expansion:
        backbone = add_block(backbone, position, drop_path_rate=block_expansion_path_dropout)

    return backbone


def add_block(backbone, position, drop_path_rate=0.0, init_value=0.0):
    '''
    Add a block to the backbone of the DINOv2 model.
    :param backbone: The backbone to expand.
    :param position: The position to add the block.
    :param drop_path_rate: Path drop out rate.
    :return: The expanded backbone.
    '''
    copy_block_position = position - 1

    added_block = copy.deepcopy(backbone.blocks[copy_block_position])

    # Initialize LayerScale parameters (ls1 and ls2) zero
    dim = added_block.ls1.gamma.size(0)  # Assuming ls1 and ls2 have the same dimension
    added_block.ls1 = LayerScale(dim, init_value)
    added_block.ls2 = LayerScale(dim, init_value)

    # set block dropout rate
    added_block.sample_drop_ratio = drop_path_rate

    added_block.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    added_block.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    expanded_backbone = copy.deepcopy(backbone)
    #add added_block one position after position
    expanded_backbone.blocks.insert(position, added_block)

    # Add the block to the backbone
    return expanded_backbone

def get_expanded_block_positions(original_positions):
    original_positions.sort()

    position_list_after_expansion = []
    for position in original_positions:
        # calclculate the position after expansion
        position_after_expansion = position + original_positions.index(position) + 1
        position_list_after_expansion.append(position_after_expansion)

    return position_list_after_expansion