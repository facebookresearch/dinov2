# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from dinov2.hub.backbones import dinov2_vitb14_scriptable, dinov2_vitg14_scriptable, dinov2_vitl14_scriptable, dinov2_vits14_scriptable
from dinov2.hub.backbones import dinov2_vitb14_reg_scriptable, dinov2_vitg14_reg_scriptable, dinov2_vitl14_reg_scriptable, dinov2_vits14_reg_scriptable
from dinov2.hub.classifiers import dinov2_vitb14_lc_scriptable, dinov2_vitg14_lc_scriptable, dinov2_vitl14_lc_scriptable, dinov2_vits14_lc_scriptable
from dinov2.hub.classifiers import dinov2_vitb14_reg_lc_scriptable, dinov2_vitg14_reg_lc_scriptable, dinov2_vitl14_reg_lc_scriptable, dinov2_vits14_reg_lc_scriptable
from dinov2.hub.depthers import dinov2_vitb14_ld_scriptable, dinov2_vitg14_ld_scriptable, dinov2_vitl14_ld_scriptable, dinov2_vits14_ld_scriptable
from dinov2.hub.depthers import dinov2_vitb14_dd_scriptable, dinov2_vitg14_dd_scriptable, dinov2_vitl14_dd_scriptable, dinov2_vits14_dd_scriptable


dependencies = ["torch"]
